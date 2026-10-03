import pytest
from cryptography.exceptions import InvalidTag
from mlkem.parameters.params import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from mlkem.crypto.hybrid import HybridPQC


@pytest.mark.parametrize("params", [ML_KEM_512, ML_KEM_768, ML_KEM_1024])
def test_hybrid_encryption_roundtrip(params):
    hybrid = HybridPQC(params)

    # 1. Generar par de claves
    ek, dk = hybrid.keygen()

    # 2. Cifrar
    original_message = b"Mensaje confidencial post-cuantico de prueba"
    payload = hybrid.encrypt_payload(ek, original_message)

    # 3. Descifrar
    decrypted = hybrid.decrypt_payload(
        dk=dk,
        capsule=payload["capsule"],
        nonce=payload["nonce"],
        ciphertext=payload["ciphertext"],
    )

    assert decrypted == original_message


def test_hybrid_tampered_ciphertext_fails():
    hybrid = HybridPQC(ML_KEM_768)
    ek, dk = hybrid.keygen()

    payload = hybrid.encrypt_payload(ek, b"Mensaje integro")

    # Modificamos un bit del texto cifrado
    tampered_ciphertext = bytearray(payload["ciphertext"])
    tampered_ciphertext[0] ^= 0xFF

    with pytest.raises(InvalidTag):
        hybrid.decrypt_payload(
            dk=dk,
            capsule=payload["capsule"],
            nonce=payload["nonce"],
            ciphertext=bytes(tampered_ciphertext),
        )


def test_hybrid_tampered_capsule_fails():
    hybrid = HybridPQC(ML_KEM_768)
    ek, dk = hybrid.keygen()

    payload = hybrid.encrypt_payload(ek, b"Mensaje integro")

    # Modificamos un bit de la capsula de Kyber
    tampered_capsule = bytearray(payload["capsule"])
    tampered_capsule[0] ^= 0xFF

    with pytest.raises(InvalidTag):
        hybrid.decrypt_payload(
            dk=dk,
            capsule=bytes(tampered_capsule),
            nonce=payload["nonce"],
            ciphertext=payload["ciphertext"],
        )