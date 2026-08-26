from cocotb.triggers import RisingEdge

ARRAY = 14

OP_LOAD_W = 0x0
OP_LOAD_B = 0x1
OP_LOAD_ACT = 0x2
OP_MATMUL = 0x3
OP_ACTIVATE = 0x4
OP_STORE = 0x5

ACT_PASSTHROUGH = 0x2

# ACTIVATE's act_scale_m occupies the TOP 17 bits of ddr3_addr (instr[55:39]),
# not the bottom. A raw multiplier value must be shifted
# left by this
ACT_SCALE_M_SHIFT = 11

WEIGHT_BASE = 0x001000
BIAS_BASE = 0x006000
ACT_BASE = 0x008000
OUT_BASE = 0x010000


def enc(opcode, acc_mode=0, target=0, ddr3_addr=0, byte_count=0, tile_params=0):
    return (
        ((opcode & 0xF) << 60)
        | ((acc_mode & 1) << 59)
        | ((target & 0x7) << 56)
        | ((ddr3_addr & 0xFFFFFFF) << 28)
        | ((byte_count & 0xFFFF) << 12)
        | (tile_params & 0xFFF)
    )


def s8(v):
    return v & 0xFF


def ref_matmul(A, W):
    M = len(A)
    N = len(W[0])
    K = len(W)
    O = [[0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            s = 0
            for k in range(K):
                s += A[m][k] * W[k][n]
            O[m][n] = s
    return O


def flatten_i32(vals):
    b = []
    for v in vals:
        b.extend([(v >> 0) & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF])
    return b


def make_tiles(total, size=ARRAY):
    tiles = []
    off = 0
    while off < total:
        sz = min(size, total - off)
        tiles.append((off, sz))
        off += sz
    return tiles


def build(
    A,
    W,
    weight_base=WEIGHT_BASE,
    bias_base=BIAS_BASE,
    act_base=ACT_BASE,
    out_base=OUT_BASE,
):
    """Return (instrs, ddr, out).

    instrs: list of 64-bit instruction words.
    ddr:    list of (addr, list_of_byte_ints) blocks to stage into DDR3.
    out:    list of (addr, list_of_signed_int8_values) expected outputs.

    The *_base params default to the module constants; a test can override
    one to place a tile at a specific address (e.g. straddling a 4KB AXI
    burst boundary).
    """
    M = len(A)
    K = len(W)
    N = len(W[0])
    O = ref_matmul(A, W)

    M_tiles = make_tiles(M)
    K_tiles = make_tiles(K)
    N_tiles = make_tiles(N)

    instrs = []
    ddr = []
    out = []

    # ---- weights ----
    waddr = weight_base
    weight_tile_addr = {}
    for ki, (ko, ks) in enumerate(K_tiles):
        for ni, (no, ns) in enumerate(N_tiles):
            rows = []
            for kk in range(ARRAY):
                for nn in range(ARRAY):
                    k = ko + kk
                    n = no + nn
                    rows.append(s8(W[k][n]) if (k < K and n < N) else 0)
            weight_tile_addr[(ki, ni)] = waddr
            ddr.append((waddr, rows))
            waddr += 200

    # ---- bias ----
    baddr = bias_base
    bias_tile_addr = {}
    for ni, (no, ns) in enumerate(N_tiles):
        vals = [0] * ARRAY
        bias_tile_addr[ni] = baddr
        ddr.append((baddr, flatten_i32(vals)))
        baddr += 56

    # ---- activations ----
    aaddr = act_base
    act_tile_addr = {}
    for mi, (mo, ms) in enumerate(M_tiles):
        for ki, (ko, ks) in enumerate(K_tiles):
            rows = []
            for mm in range(ms):
                for kk in range(ARRAY):
                    k = ko + kk
                    rows.append(s8(A[mo + mm][k]) if k < K else 0)
            act_tile_addr[(mi, ki)] = aaddr
            ddr.append((aaddr, rows))
            aaddr += 200

    # ---- instruction stream + expected output ----
    oaddr = out_base
    for ni, (no, ns) in enumerate(N_tiles):
        for mi, (mo, ms) in enumerate(M_tiles):
            for ki, (ko, ks) in enumerate(K_tiles):
                instrs.append(
                    enc(
                        OP_LOAD_W,
                        ddr3_addr=weight_tile_addr[(ki, ni)],
                        byte_count=14 * ARRAY,
                    )
                )
                if ki == 0:
                    instrs.append(
                        enc(
                            OP_LOAD_B,
                            ddr3_addr=bias_tile_addr[ni],
                            byte_count=ARRAY * 4,
                        )
                    )
                instrs.append(
                    enc(
                        OP_LOAD_ACT,
                        ddr3_addr=act_tile_addr[(mi, ki)],
                        byte_count=ms * ARRAY,
                    )
                )
                instrs.append(
                    enc(OP_MATMUL, acc_mode=(0 if ki == 0 else 1), tile_params=ms)
                )
            instrs.append(
                enc(
                    OP_ACTIVATE,
                    target=ACT_PASSTHROUGH,
                    ddr3_addr=1 << ACT_SCALE_M_SHIFT,
                    byte_count=0,
                    tile_params=ms,
                )
            )
            instrs.append(enc(OP_STORE, ddr3_addr=oaddr, byte_count=ms * ARRAY))

            exp = []
            for mm in range(ms):
                for nn in range(ARRAY):
                    m = mo + mm
                    n = no + nn
                    exp.append(O[m][n] if n < N else 0)
            out.append((oaddr, exp))
            oaddr += 200

    return instrs, ddr, out


class AxiRam:
    """AXI4 burst-capable slave (DDR3 stand-in) backed by a byte dict.

    Honors AWLEN/ARLEN: serves ARLEN+1 sequential 8-byte beats per read
    burst (RLAST on the last one) and accepts AWLEN+1 beats per write burst
    before issuing a single B response - matching dma_unit's burst master
    (hw/rtl/control/dma_unit.sv) and real AXI4/MIG semantics. The arbiter
    (axi_arbiter.sv) only ever forwards one complete burst at a time from
    either master, so bursts here are never interleaved.
    """

    def __init__(self, dut):
        self.dut = dut
        self.mem = {}
        # Read burst state
        self._r_addr = 0
        self._r_beats_left = 0
        # Write burst state
        self._w_addr = 0
        self._w_beats_left = 0

    def write_bytes(self, addr, data):
        for i, b in enumerate(data):
            self.mem[addr + i] = b & 0xFF

    def read_bytes(self, addr, n):
        return [self.mem.get(addr + i, 0) for i in range(n)]

    @staticmethod
    def _check_4k(addr, beats, kind):
        """AXI4 forbids a single burst from crossing a 4KB address boundary
        (real slaves are not required to handle it correctly). dma_unit
        clamps its burst length to avoid this - verify it actually does.
        """
        span = beats * 8
        first_page = addr // 4096
        last_page = (addr + span - 1) // 4096
        assert first_page == last_page, (
            f"AXI4 4KB boundary violation: {kind} burst at addr=0x{addr:x} "
            f"beats={beats} spans 0x{addr:x}..0x{addr + span - 1:x}, "
            f"crossing a 4096-byte page boundary"
        )

    def init_signals(self):
        d = self.dut
        d.m_axi_awready.value = 0
        d.m_axi_wready.value = 0
        d.m_axi_bvalid.value = 0
        d.m_axi_bresp.value = 0
        d.m_axi_arready.value = 0
        d.m_axi_rvalid.value = 0
        d.m_axi_rresp.value = 0
        d.m_axi_rdata.value = 0
        d.m_axi_rlast.value = 0

    async def run(self):
        """Serve burst AXI4 transactions forever."""
        d = self.dut
        self.init_signals()

        while True:
            await RisingEdge(d.clk)

            # ---- read address channel: accept a new burst when idle ----
            d.m_axi_arready.value = 0
            if self._r_beats_left == 0 and int(d.m_axi_arvalid.value):
                d.m_axi_arready.value = 1
                self._r_addr = int(d.m_axi_araddr.value)
                self._r_beats_left = int(d.m_axi_arlen.value) + 1
                self._check_4k(self._r_addr, self._r_beats_left, "read")

            # ---- read data channel (1-cycle latency), one beat per cycle ----
            d.m_axi_rvalid.value = 0
            if self._r_beats_left > 0:
                val = 0
                for i in range(8):
                    val |= self.mem.get(self._r_addr + i, 0) << (8 * i)
                d.m_axi_rdata.value = val
                d.m_axi_rvalid.value = 1
                d.m_axi_rlast.value = 1 if self._r_beats_left == 1 else 0
                if int(d.m_axi_rready.value):
                    self._r_addr += 8
                    self._r_beats_left -= 1

            # ---- write address channel: accept a new burst when idle ----
            d.m_axi_awready.value = 0
            if self._w_beats_left == 0 and int(d.m_axi_awvalid.value):
                d.m_axi_awready.value = 1
                self._w_addr = int(d.m_axi_awaddr.value)
                self._w_beats_left = int(d.m_axi_awlen.value) + 1
                self._check_4k(self._w_addr, self._w_beats_left, "write")

            # ---- write data channel, one beat per cycle ----
            d.m_axi_wready.value = 0
            if self._w_beats_left > 0:
                d.m_axi_wready.value = 1
                if int(d.m_axi_wvalid.value):
                    data = int(d.m_axi_wdata.value)
                    for i in range(8):
                        self.mem[self._w_addr + i] = (data >> (8 * i)) & 0xFF
                    self._w_addr += 8
                    self._w_beats_left -= 1
                    if self._w_beats_left == 0:
                        d.m_axi_bvalid.value = 1

            # ---- write response channel ----
            if int(d.m_axi_bvalid.value) and int(d.m_axi_bready.value):
                d.m_axi_bvalid.value = 0
