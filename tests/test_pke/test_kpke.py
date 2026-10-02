import os
import pytest
from mlkem.parameters.params import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from mlkem.pke.kpke import k_pke_keygen, k_pke_encrypt, k_pke_decrypt


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_kpke_key_and_ciphertext_sizes(params):
    """Valida los tamaños exactos de ek, dk y ciphertext con la Tabla 3 de FIPS 203."""
    d = os.urandom(32)
    m = os.urandom(32)
    r = os.urandom(32)

    ek, dk = k_pke_keygen(params, d)
    c = k_pke_encrypt(params, ek, m, r)

    assert len(ek) == 384 * params.k + 32
    assert len(dk) == 384 * params.k
    assert len(c) == 32 * (params.du * params.k + params.dv)


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_kpke_encrypt_decrypt_roundtrip(params):
    """Prueba que Decrypt(Encrypt(m)) == m en múltiples ejecuciones."""
    for _ in range(5):
        d = os.urandom(32)
        m = os.urandom(32)
        r = os.urandom(32)

        ek, dk = k_pke_keygen(params, d)
        c = k_pke_encrypt(params, ek, m, r)
        m_recovered = k_pke_decrypt(params, dk, c)

        assert m_recovered == m, f"Fallo al descifrar mensaje en {params.name}"


def test_kpke_deterministic_generation():
    """Valida que semillas idénticas generen exactamente las mismas salidas."""
    params = ML_KEM_768
    d = b"\x01" * 32
    m = b"\x02" * 32
    r = b"\x03" * 32

    ek1, dk1 = k_pke_keygen(params, d)
    ek2, dk2 = k_pke_keygen(params, d)
    assert ek1 == ek2
    assert dk1 == dk2

    c1 = k_pke_encrypt(params, ek1, m, r)
    c2 = k_pke_encrypt(params, ek1, m, r)
    assert c1 == c2


def test_kpke_ciphertext_tampering():
    """Modificar un byte del ciphertext debe impedir recuperar el mensaje original."""
    params = ML_KEM_768
    d = os.urandom(32)
    m = os.urandom(32)
    r = os.urandom(32)

    ek, dk = k_pke_keygen(params, d)
    c = bytearray(k_pke_encrypt(params, ek, m, r))

    # Corrompemos un byte en la sección v (últimos bytes del ciphertext)
    c[-1] ^= 0xFF

    m_recovered = k_pke_decrypt(params, dk, bytes(c))
    assert m_recovered != m


def test_kpke_invalid_input_lengths():
    params = ML_KEM_768
    valid_seed = os.urandom(32)
    ek, dk = k_pke_keygen(params, valid_seed)

    with pytest.raises(ValueError):
        k_pke_keygen(params, b"\x00" * 31)

    with pytest.raises(ValueError):
        k_pke_encrypt(params, ek, b"\x00" * 31, valid_seed)  # m corto

    with pytest.raises(ValueError):
        k_pke_decrypt(params, dk, b"\x00" * 100)  # c de tamaño erróneo