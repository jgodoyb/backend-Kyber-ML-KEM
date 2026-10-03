"""Suite de pruebas para la API REST simplificada Q-Proof Kyber ML-KEM.

Verifica:
1. Endpoint de estado y diagnóstico (GET /).
2. Generación de claves ML-KEM (POST /keygen) con valores por defecto y niveles FIPS 203.
3. Flujo completo de Cifrado Híbrido (POST /keygen -> POST /encrypt -> POST /decrypt).
4. Fallos de integridad y autenticación (ciphertext, cápsula o nonce manipulados).
5. Validaciones de entrada (Base64 inválido, claves corruptas y niveles de seguridad no permitidos).
6. Rate limiting con SlowAPI (HTTP 429 tras superar la cuota permitida).
"""

import base64
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def disable_rate_limiter_for_unit_tests():
    """Desactiva temporalmente el rate limiter para permitir la ejecución rápida de tests."""
    app.state.limiter.enabled = False
    yield
    app.state.limiter.enabled = False


@pytest.fixture
def client():
    """Cliente de pruebas HTTP sincronizado con FastAPI."""
    return TestClient(app)


# ==============================================================================
# 1. Diagnóstico y Salud de la API
# ==============================================================================

def test_health_check_endpoint(client):
    """Verifica que GET / responde exitosamente y expone el estado online del servicio."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "Q-Proof Kyber ML-KEM API"
    assert 768 in data["supported_levels"]


# ==============================================================================
# 2. Generación de Claves (POST /keygen)
# ==============================================================================

def test_keygen_default_security_level(client):
    """Verifica que al no especificar security_level se utilice por defecto 768."""
    keygen_res = client.post("/keygen", json={})
    assert keygen_res.status_code == 200
    data = keygen_res.json()
    assert data["security_level"] == 768
    assert "ek_b64" in data
    assert "dk_b64" in data


@pytest.mark.parametrize("security_level", [512, 768, 1024])
def test_keygen_all_security_levels(client, security_level):
    """Verifica la generación de pares de claves para todos los niveles aprobados por FIPS 203."""
    keygen_res = client.post("/keygen", json={"security_level": security_level})
    assert keygen_res.status_code == 200
    data = keygen_res.json()
    assert data["security_level"] == security_level

    raw_ek = base64.b64decode(data["ek_b64"])
    raw_dk = base64.b64decode(data["dk_b64"])
    assert len(raw_ek) > 0
    assert len(raw_dk) > 0


# ==============================================================================
# 3. Flujo Completo Cifrado Híbrido (Keygen -> Encrypt -> Decrypt)
# ==============================================================================

@pytest.mark.parametrize("security_level", [512, 768, 1024])
@pytest.mark.parametrize(
    "plaintext",
    [
        "Texto confidencial con caracteres especiales: áéíóú ñ ¿¡ 🚀 Q-Proof",
        "A" * 500,
        "Payload con formato Base64: SGVsbG8gV29ybGQgZnJvbSBQSUMh",
    ],
)
def test_hybrid_encryption_full_flow(client, security_level, plaintext):
    """Verifica que el cifrado y descifrado híbrido recupere el mensaje original íntegro."""
    # 1. Generar claves
    keygen_res = client.post("/keygen", json={"security_level": security_level})
    assert keygen_res.status_code == 200
    keys = keygen_res.json()

    # 2. Cifrar (Encrypt)
    encrypt_res = client.post(
        "/encrypt",
        json={
            "ek_b64": keys["ek_b64"],
            "plaintext": plaintext,
            "security_level": security_level,
        },
    )
    assert encrypt_res.status_code == 200
    cipher_payload = encrypt_res.json()

    assert "capsule_b64" in cipher_payload
    assert "nonce_b64" in cipher_payload
    assert "ciphertext_b64" in cipher_payload

    # 3. Descifrar (Decrypt)
    decrypt_res = client.post(
        "/decrypt",
        json={
            "dk_b64": keys["dk_b64"],
            "capsule_b64": cipher_payload["capsule_b64"],
            "nonce_b64": cipher_payload["nonce_b64"],
            "ciphertext_b64": cipher_payload["ciphertext_b64"],
            "security_level": security_level,
        },
    )
    assert decrypt_res.status_code == 200
    decrypt_data = decrypt_res.json()

    assert "plaintext" in decrypt_data
    assert decrypt_data["plaintext"] == plaintext


# ==============================================================================
# 4. Manejo de Errores e Integridad Criptográfica (HTTP 400 y 422)
# ==============================================================================

def test_tampered_ciphertext_fails_integrity(client):
    """Verifica que alterar el texto cifrado resulte en HTTP 400 con mensaje de fallo de integridad."""
    keygen = client.post("/keygen", json={"security_level": 768}).json()
    encrypted = client.post(
        "/encrypt",
        json={"ek_b64": keygen["ek_b64"], "plaintext": "Valid Message", "security_level": 768},
    ).json()

    # Manipular un byte del ciphertext
    raw_ct = bytearray(base64.b64decode(encrypted["ciphertext_b64"]))
    raw_ct[0] ^= 0xFF
    tampered_ct_b64 = base64.b64encode(raw_ct).decode("ascii")

    res = client.post(
        "/decrypt",
        json={
            "dk_b64": keygen["dk_b64"],
            "capsule_b64": encrypted["capsule_b64"],
            "nonce_b64": encrypted["nonce_b64"],
            "ciphertext_b64": tampered_ct_b64,
            "security_level": 768,
        },
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "Decryption failed: integrity check failed"


def test_tampered_capsule_fails_integrity(client):
    """Verifica que alterar la cápsula ML-KEM cause fallo de autenticación en AES-GCM (HTTP 400)."""
    keygen = client.post("/keygen", json={"security_level": 768}).json()
    encrypted = client.post(
        "/encrypt",
        json={"ek_b64": keygen["ek_b64"], "plaintext": "Valid Message", "security_level": 768},
    ).json()

    # Manipular un byte de la cápsula
    raw_capsule = bytearray(base64.b64decode(encrypted["capsule_b64"]))
    raw_capsule[0] ^= 0xFF
    tampered_capsule_b64 = base64.b64encode(raw_capsule).decode("ascii")

    res = client.post(
        "/decrypt",
        json={
            "dk_b64": keygen["dk_b64"],
            "capsule_b64": tampered_capsule_b64,
            "nonce_b64": encrypted["nonce_b64"],
            "ciphertext_b64": encrypted["ciphertext_b64"],
            "security_level": 768,
        },
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "Decryption failed: integrity check failed"


def test_tampered_nonce_fails_integrity(client):
    """Verifica que un nonce manipulado impida el descifrado autenticado (HTTP 400)."""
    keygen = client.post("/keygen", json={"security_level": 768}).json()
    encrypted = client.post(
        "/encrypt",
        json={"ek_b64": keygen["ek_b64"], "plaintext": "Valid Message", "security_level": 768},
    ).json()

    # Manipular un byte del nonce
    raw_nonce = bytearray(base64.b64decode(encrypted["nonce_b64"]))
    raw_nonce[0] ^= 0xFF
    tampered_nonce_b64 = base64.b64encode(raw_nonce).decode("ascii")

    res = client.post(
        "/decrypt",
        json={
            "dk_b64": keygen["dk_b64"],
            "capsule_b64": encrypted["capsule_b64"],
            "nonce_b64": tampered_nonce_b64,
            "ciphertext_b64": encrypted["ciphertext_b64"],
            "security_level": 768,
        },
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "Decryption failed: integrity check failed"


def test_invalid_base64_returns_422_or_400(client):
    """Verifica que cadenas Base64 sintácticamente incorrectas sean rechazadas con HTTP 422 o 400."""
    invalid_b64 = "!!!not-valid-base64!!!"

    # 1. En cifrado (ek_b64 mal formado)
    res_enc = client.post(
        "/encrypt",
        json={"ek_b64": invalid_b64, "plaintext": "hello", "security_level": 768},
    )
    assert res_enc.status_code in (400, 422)

    # 2. En descifrado (campos mal formados)
    res_dec_dk = client.post(
        "/decrypt",
        json={
            "dk_b64": invalid_b64,
            "capsule_b64": "AAAA",
            "nonce_b64": "AAAA",
            "ciphertext_b64": "AAAA",
            "security_level": 768,
        },
    )
    assert res_dec_dk.status_code in (400, 422)

    res_dec_nonce = client.post(
        "/decrypt",
        json={
            "dk_b64": "AAAA",
            "capsule_b64": "AAAA",
            "nonce_b64": invalid_b64,
            "ciphertext_b64": "AAAA",
            "security_level": 768,
        },
    )
    assert res_dec_nonce.status_code in (400, 422)


def test_corrupted_key_length_returns_400_or_422(client):
    """Verifica que una clave con longitud incorrecta sea rechazada con HTTP 400 o 422."""
    truncated_key_b64 = base64.b64encode(b"\x00" * 20).decode("ascii")

    res_enc = client.post(
        "/encrypt",
        json={"ek_b64": truncated_key_b64, "plaintext": "test", "security_level": 768},
    )
    assert res_enc.status_code in (400, 422)

    res_dec = client.post(
        "/decrypt",
        json={
            "dk_b64": truncated_key_b64,
            "capsule_b64": truncated_key_b64,
            "nonce_b64": base64.b64encode(b"\x00" * 12).decode("ascii"),
            "ciphertext_b64": base64.b64encode(b"\x00" * 16).decode("ascii"),
            "security_level": 768,
        },
    )
    assert res_dec.status_code in (400, 422)


def test_invalid_security_level_returns_422(client):
    """Verifica que niveles de seguridad no permitidos por FIPS 203 sean rechazados con 422."""
    res_keygen = client.post("/keygen", json={"security_level": 999})
    assert res_keygen.status_code == 422

    res_enc = client.post(
        "/encrypt",
        json={"ek_b64": "AAAA", "plaintext": "test", "security_level": 256},
    )
    assert res_enc.status_code == 422


# ==============================================================================
# 5. Rate Limiting (SlowAPI)
# ==============================================================================

def test_rate_limiting_slowapi(client):
    """Verifica que el decorador de slowapi limite a 10 req/min y devuelva HTTP 429 al excederlo."""
    try:
        app.state.limiter.enabled = True
        app.state.limiter.reset()

        # Realizar 10 peticiones exitosas consecutivas
        for _ in range(10):
            res = client.post("/keygen", json={"security_level": 768})
            assert res.status_code == 200

        # La 11ª petición dentro del mismo minuto debe responder 429 Too Many Requests
        res_blocked = client.post("/keygen", json={"security_level": 768})
        assert res_blocked.status_code == 429
    finally:
        app.state.limiter.enabled = False
