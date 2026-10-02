import pytest
from mlkem.constants import N, Q
from mlkem.encoding.coding import compress, decompress, byte_encode, byte_decode


def test_compress_decompress_identity():
    """FIPS 203 Sección 4.2.1:

    Para todo d < 12 y todo y in Z_{2^d}, Compress_d(Decompress_d(y)) == y.
    """
    for d in [1, 4, 10, 11]:
        mod_d = 1 << d
        step = 1 if mod_d <= 512 else (mod_d // 256)
        for y in range(0, mod_d, step):
            dec = decompress(y, d)
            rec = compress(dec, d)
            assert rec == y, f"Fallo de identidad para d={d}, y={y}"


def test_compress_error_bound():
    """FIPS 203 Sección 4.2.1:

    Comprueba que el error de compresión modular esté acotado.
    """
    for d in [4, 10]:
        for x in [0, 1, 100, 1500, 3328]:
            c = compress(x, d)
            dec = decompress(c, d)
            diff = (dec - x) % Q
            signed_diff = diff if diff <= Q // 2 else diff - Q
            max_err = (Q + (1 << (d + 1)) - 1) // (1 << (d + 1))
            assert abs(signed_diff) <= max_err + 1


def test_byte_encode_decode_roundtrip_all_d():
    """Verifica que ByteDecode_d(ByteEncode_d(F)) == F para todo 1 <= d <=

    11.
    """
    for d in range(1, 12):
        mod_d = 1 << d
        F = [(i * 37 + 13) % mod_d for i in range(N)]
        encoded = byte_encode(F, d)
        assert len(encoded) == 32 * d
        decoded = byte_decode(encoded, d)
        assert decoded == F


def test_byte_encode_decode_d12():
    """Para d=12, los coeficientes en F deben ser estrictamente menores que q

    para garantizar roundtrip biyectivo.
    """
    d = 12
    F = [(i * 73 + 5) % Q for i in range(N)]
    encoded = byte_encode(F, d)
    assert len(encoded) == 32 * 12  # 384 bytes
    decoded = byte_decode(encoded, d)
    assert decoded == F


def test_byte_decode_d12_reduces_mod_q():
    """FIPS 203: Si un entero de 12 bits decodificado está entre 3329 y 4095,

    ByteDecode_12 debe reducir el valor mod q.
    """
    # 4000 = 0x0FA0 -> little-endian en 12 bits: 0xA0 (byte 0), 0x0F (byte 1 bits bajos)
    raw = bytearray(384)
    raw[0] = 0xA0
    raw[1] = 0x0F
    decoded = byte_decode(bytes(raw), d=12)
    assert decoded[0] == 4000 % Q


def test_invalid_sizes():
    with pytest.raises(ValueError):
        byte_encode([0] * 100, d=4)

    with pytest.raises(ValueError):
        byte_decode(b"\x00" * 30, d=4)