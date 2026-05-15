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


def load_weight(ser, weight: int):
    ser.write(bytes([0x01, weight & 0xFF]))


def send_activation(ser, act: int):
    ser.write(bytes([0x02, act & 0xFF]))


def clear_acc(ser):
    ser.write(bytes([0x04]))


def read_result(ser) -> int:
    ser.write(bytes([0x03]))
    raw = ser.read(4)
    if len(raw) != 4:
        raise TimeoutError(f"Expected 4 bytes, got {len(raw)}")
    return struct.unpack("<i", raw)[0]


def stream_activations(ser, activations: list[int]):
    for act in activations:
        send_activation(ser, act)
    for _ in range(PIPE_DEPTH):
        send_activation(ser, 0)


def test_single_mac(ser):
    """weight=3, activation=4 → 12"""
    clear_acc(ser)
    load_weight(ser, 3)
    stream_activations(ser, [4])
    assert read_result(ser) == 12


def test_dot_product(ser):
    """weight=3, activations=[1..14] → 315"""
    clear_acc(ser)
    load_weight(ser, 3)
    activations = list(range(1, 15))
    expected = sum(3 * a for a in activations)
    stream_activations(ser, activations)
    assert read_result(ser) == expected


def test_signed_negative_weight(ser):
    """weight=-5, activation=7 → -35"""
    clear_acc(ser)
    load_weight(ser, -5)
    stream_activations(ser, [7])
    assert read_result(ser) == -35


def test_signed_both_negative(ser):
    """weight=-3, activation=-4 → 12"""
    clear_acc(ser)
    load_weight(ser, -3)
    stream_activations(ser, [-4])
    assert read_result(ser) == 12


def test_clear_accumulator(ser):
    """Accumulate 50, clear, verify 0, then accumulate 30"""
    clear_acc(ser)
    load_weight(ser, 10)
    stream_activations(ser, [5])
    assert read_result(ser) == 50, "pre-clear value wrong"

    clear_acc(ser)
    time.sleep(0.001)
    assert read_result(ser) == 0, "accumulator not cleared"

    stream_activations(ser, [3])
    assert read_result(ser) == 30, "re-accumulation wrong"


def test_max_positive(ser):
    """127 * 127 * 14 = 225778 — worst-case positive, well within int32"""
    clear_acc(ser)
    load_weight(ser, 127)
    stream_activations(ser, [127] * 14)
    expected = 127 * 127 * 14
    assert read_result(ser) == expected


def test_max_negative(ser):
    """-128 * 127 * 14 = -227328 — worst-case negative"""
    clear_acc(ser)
    load_weight(ser, -128)
    stream_activations(ser, [127] * 14)
    expected = -128 * 127 * 14
    assert read_result(ser) == expected


@pytest.mark.parametrize("trial", range(100))
def test_random_dot_product(ser, trial):
    rng = random.Random(trial)  # seeded so failures are reproducible
    weight = rng.randint(-128, 127)
    activations = [rng.randint(-128, 127) for _ in range(14)]
    expected = sum(weight * a for a in activations)

    clear_acc(ser)
    load_weight(ser, weight)
    stream_activations(ser, activations)

    result = read_result(ser)
    assert result == expected, (
        f"trial={trial} weight={weight} activations={activations} "
        f"expected={expected} got={result}"
    )
