import pytest
from mlkem import (
    ml_kem_keygen,
    ml_kem_encaps,
    ml_kem_decaps,
    ML_KEM_512,
    ML_KEM_768,
    ML_KEM_1024
)


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_end_to_end_key_exchange(params):
    """Prueba el ciclo completo de la API pública para todos los parámetros aprobados."""
    # Alice genera claves
    ek, dk = ml_kem_keygen(params)

    # Bob encapsula y genera la llave compartida
    k_bob, ciphertext = ml_kem_encaps(params, ek)

    # Alice decapsula y recupera la llave compartida
    k_alice = ml_kem_decaps(params, dk, ciphertext)

    assert len(k_bob) == 32
    assert len(k_alice) == 32
    assert k_bob == k_alice


def test_encaps_rejects_malformed_public_key_length():
    """Valida que Encaps falle ante longitudes inválidas."""
    with pytest.raises(ValueError, match="Clave de encapsulación inválida"):
        ml_kem_encaps(ML_KEM_768, b"\x00" * 100)


def test_encaps_rejects_modulus_check_violation():
    """Valida el modulus check:

    Si los primeros 12 bits representan un entero >= 3329 (por ejemplo 4000),
    el modulus check debe fallar.
    """
    ek, _ = ml_kem_keygen(ML_KEM_768)
    corrupted_ek = bytearray(ek)

    # 4000 = 0x0FA0 (en little endian: 0xA0 en byte 0, 0x0F en byte 1)
    corrupted_ek[0] = 0xA0
    corrupted_ek[1] = 0x0F

    with pytest.raises(ValueError, match="Clave de encapsulación inválida"):
        ml_kem_encaps(ML_KEM_768, bytes(corrupted_ek))


def test_decaps_rejects_tampered_dk_hash():
    """Valida que Decaps falle si el hash H(ek) dentro de dk no coincide."""
    params = ML_KEM_768
    ek, dk = ml_kem_keygen(params)
    _, c = ml_kem_encaps(params, ek)

    corrupted_dk = bytearray(dk)
    # Corrompemos el segmento donde reside H(ek): [768k+32 : 768k+64]
    hash_offset = 768 * params.k + 32
    corrupted_dk[hash_offset] ^= 0xFF

    with pytest.raises(ValueError, match="Entrada de desencapsulado inválida"):
        ml_kem_decaps(params, bytes(corrupted_dk), c)