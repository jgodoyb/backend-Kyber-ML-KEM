import pytest
from mlkem.primitives.conversions import bits_to_bytes, bytes_to_bits


def test_nist_example_139():
    """Valida el ejemplo explícito del documento FIPS 203 (Sección 4.2.1):

    El byte 139 corresponde a los bits 11010001 (little-endian: 1 + 2 + 8 +
    128).
    """
    expected_bits = [1, 1, 0, 1, 0, 0, 0, 1]
    byte_val = bytes([139])

    # BytesToBits
    bits = bytes_to_bits(byte_val)
    assert bits == expected_bits

    # BitsToBytes
    reconstructed_bytes = bits_to_bytes(expected_bits)
    assert reconstructed_bytes == byte_val


def test_roundtrip_all_single_bytes():
    """Comprueba que para todos los valores posibles de un byte (0 a 255),

    la ida y vuelta conserve la identidad exacta.
    """
    for val in range(256):
        original = bytes([val])
        bits = bytes_to_bits(original)
        assert len(bits) == 8
        recovered = bits_to_bytes(bits)
        assert recovered == original


def test_roundtrip_multibyte():
    """Prueba secuencias de varios bytes consecutivos."""
    data = bytes([0x00, 0xFF, 0x55, 0xAA, 0x12, 0x34])
    bits = bytes_to_bits(data)
    assert len(bits) == len(data) * 8
    assert bits_to_bytes(bits) == data


def test_empty_input():
    """Comprueba el comportamiento ante arrays vacíos."""
    assert bytes_to_bits(b"") == []
    assert bits_to_bytes([]) == b""


def test_invalid_bit_length():
    """Valida que BitsToBytes falle si la longitud no es múltiplo de 8."""
    with pytest.raises(ValueError):
        bits_to_bytes([1, 0, 1])


def test_invalid_bit_values():
    """Valida que BitsToBytes falle si algún bit no es 0 o 1."""
    with pytest.raises(ValueError):
        bits_to_bytes([0, 1, 2, 0, 0, 0, 0, 0])