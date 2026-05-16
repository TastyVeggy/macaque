import serial
import struct
import random
import time
import pytest

PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
PIPE_DEPTH = 3


@pytest.fixture(scope="session")
def ser():
    s = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(0.1)
    yield s
    s.close()


def stream(ser, activations: list[int], accumulations: list[int]):
    assert len(activations) == len(accumulations)
    for act, acc in zip(activations, accumulations):
        send_pe_input(ser, act, acc)
    # No flush — PE overwrites P each cycle, zeros would destroy the result.
    # UART is slow enough that the pipeline has settled before read_result() fires.
    time.sleep(0.005)


def load_weight(ser, weight: int):
    ser.write(bytes([0x01, weight & 0xFF]))


def send_pe_input(ser, act: int, acc: int):
    acc_bytes = struct.pack("<i", acc)
    ser.write(bytes([0x02, act & 0xFF]) + acc_bytes)


def read_result(ser) -> int:
    ser.write(bytes([0x03]))
    raw = ser.read(4)
    if len(raw) != 4:
        raise TimeoutError(f"Expected 4 bytes, got {len(raw)}")
    return struct.unpack("<i", raw)[0]


def test_single_multiply(ser):
    """weight=3, act=4, acc_in=0 → 12"""
    load_weight(ser, 3)
    stream(ser, [4], [0])
    assert read_result(ser) == 12


def test_single_mac(ser):
    """weight=3, act=4, acc_in=100 → 112"""
    load_weight(ser, 3)
    stream(ser, [4], [100])
    assert read_result(ser) == 112


def test_simulated_column(ser):
    """Simulate a column of 3 PEs by chaining acc_out into acc_in.
    weight=3, acts=[1,2,4], acc_ins=[0,3,9] → final acc_out=21"""
    load_weight(ser, 3)

    acts = [1, 2, 4]
    acc_ins = [0, 3, 9]

    stream(ser, acts, acc_ins)
    result = read_result(ser)
    assert result == 21, f"Expected 21, got {result}"


def test_signed_negative_weight(ser):
    """weight=-5, act=7, acc_in=0 → -35"""
    load_weight(ser, -5)
    stream(ser, [7], [0])
    assert read_result(ser) == -35


def test_signed_both_negative(ser):
    """weight=-3, act=-4, acc_in=0 → 12"""
    load_weight(ser, -3)
    stream(ser, [-4], [0])
    assert read_result(ser) == 12


def test_negative_acc_in(ser):
    """weight=3, act=4, acc_in=-100 → -88"""
    load_weight(ser, 3)
    stream(ser, [4], [-100])
    assert read_result(ser) == -88


def test_acc_in_dominates(ser):
    """Large acc_in: weight=1, act=1, acc_in=10000 → 10001"""
    load_weight(ser, 1)
    stream(ser, [1], [10000])
    assert read_result(ser) == 10001


def test_acc_in_passthrough(ser):
    """weight=1, act=0, acc_in=42 → 42 (pure passthrough)"""
    load_weight(ser, 1)
    stream(ser, [0], [42])
    assert read_result(ser) == 42


def test_max_positive(ser):
    """weight=127, act=127, acc_in=0 → 16129"""
    load_weight(ser, 127)
    stream(ser, [127], [0])
    assert read_result(ser) == 16129


def test_max_negative(ser):
    """weight=-128, act=127, acc_in=0 → -16256"""
    load_weight(ser, -128)
    stream(ser, [127], [0])
    assert read_result(ser) == -16256


def test_weight_change(ser):
    """Verify weight register updates correctly"""
    load_weight(ser, 3)
    stream(ser, [4], [0])
    assert read_result(ser) == 12

    load_weight(ser, 7)
    stream(ser, [4], [0])
    assert read_result(ser) == 28


@pytest.mark.parametrize("trial", range(100))
def test_random_pe(ser, trial):
    rng = random.Random(trial)
    weight = rng.randint(-128, 127)
    act = rng.randint(-128, 127)
    acc = rng.randint(-(2**31), 2**31 - 1)
    expected = acc + weight * act

    load_weight(ser, weight)
    stream(ser, [act], [acc])
    result = read_result(ser)
    assert result == expected, (
        f"trial={trial} weight={weight} act={act} acc_in={acc} "
        f"expected={expected} got={result}"
    )
