import os
import pytest
from mlkem.parameters.params import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from mlkem.crypto.hash import hash_j
from mlkem.core.internal import (
    ml_kem_keygen_internal,
    ml_kem_encaps_internal,
    ml_kem_decaps_internal
)


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_ml_kem_internal_key_agreement(params):
    """Verifica que K == K' para todos los perfiles oficiales del FIPS 203."""
    for _ in range(3):
        d = os.urandom(32)
        z = os.urandom(32)
        m = os.urandom(32)

        ek, dk = ml_kem_keygen_internal(params, d, z)
        K_alice, c = ml_kem_encaps_internal(params, ek, m)
        K_bob = ml_kem_decaps_internal(params, dk, c)

        assert len(K_alice) == 32
        assert K_alice == K_bob, f"Fallo de coincidencia en el secreto compartido ({params.name})"


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_ml_kem_internal_key_sizes(params):
    """Comprueba las longitudes de ek y dk especificadas en la Tabla 3."""
    d = os.urandom(32)
    z = os.urandom(32)
    ek, dk = ml_kem_keygen_internal(params, d, z)

    expected_ek_len = 384 * params.k + 32
    expected_dk_len = 768 * params.k + 96

    assert len(ek) == expected_ek_len
    assert len(dk) == expected_dk_len


def test_ml_kem_implicit_rejection():
    """Valida el mecanismo de rechazo implícito (Fujisaki-Okamoto):

    Si c es manipulado, Decaps_internal debe retornar J(z || c) sin abortar.
    """
    params = ML_KEM_768
    d = os.urandom(32)
    z = os.urandom(32)
    m = os.urandom(32)

    ek, dk = ml_kem_keygen_internal(params, d, z)
    K_valid, c = ml_kem_encaps_internal(params, ek, m)

    # Alteramos el ciphertext
    corrupted_c = bytearray(c)
    corrupted_c[0] ^= 0x01
    corrupted_c = bytes(corrupted_c)

    # Al desencapsular un ciphertext corrupto:
    K_rejected = ml_kem_decaps_internal(params, dk, corrupted_c)

    # Debe ser diferente a la clave legítima
    assert K_rejected != K_valid

    # Debe ser exactamente J(z || corrupted_c)
    expected_rejected_key = hash_j(z + corrupted_c)
    assert K_rejected == expected_rejected_key


def test_ml_kem_internal_deterministic():
    """Valida que entradas idénticas produzcan salidas idénticas."""
    params = ML_KEM_768
    d = b"\x10" * 32
    z = b"\x20" * 32
    m = b"\x30" * 32

    ek1, dk1 = ml_kem_keygen_internal(params, d, z)
    ek2, dk2 = ml_kem_keygen_internal(params, d, z)
    assert ek1 == ek2
    assert dk1 == dk2

    K1, c1 = ml_kem_encaps_internal(params, ek1, m)
    K2, c2 = ml_kem_encaps_internal(params, ek1, m)
    assert K1 == K2
    assert c1 == c2


def test_invalid_lengths():
    params = ML_KEM_768
    with pytest.raises(ValueError):
        ml_kem_keygen_internal(params, b"\x00" * 31, b"\x00" * 32)

    with pytest.raises(ValueError):
        ml_kem_encaps_internal(params, b"\x00" * 1184, b"\x00" * 31)

    with pytest.raises(ValueError):
        ml_kem_decaps_internal(params, b"\x00" * 100, b"\x00" * 1088)