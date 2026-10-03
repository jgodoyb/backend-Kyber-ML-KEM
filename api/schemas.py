"""Esquemas Pydantic para la API REST Q-Proof Kyber ML-KEM.

Define los modelos de solicitud y respuesta para:
- Generación de claves ML-KEM (FIPS 203)
- Cifrado y descifrado híbrido autenticado (ML-KEM + AES-256-GCM)
"""

import base64
import binascii
from typing import Literal
from pydantic import BaseModel, Field, field_validator

# Niveles de seguridad aprobados por FIPS 203 (NIST)
SecurityLevel = Literal[512, 768, 1024]


def _validate_base64_field(value: str, field_name: str) -> str:
    """Valida que una cadena represente un Base64 sintácticamente correcto."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} debe ser una cadena de texto.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} no puede estar vacío.")
    try:
        # validate=True verifica caracteres del alfabeto y padding correcto
        base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Codificación Base64 inválida para '{field_name}': {exc}")
    return stripped


# ==============================================================================
# Esquemas: Generación de Claves (Keygen)
# ==============================================================================

class KeygenRequest(BaseModel):
    """Solicitud de generación de pares de claves ML-KEM."""

    security_level: SecurityLevel = Field(
        default=768,
        description="Nivel de seguridad NIST ML-KEM: 512 (Cat 1), 768 (Cat 3), 1024 (Cat 5).",
        examples=[768],
    )


class KeygenResponse(BaseModel):
    """Respuesta con el par de claves generado en Base64."""

    ek_b64: str = Field(
        ...,
        description="Clave pública de encapsulación (ek) codificada en Base64.",
    )
    dk_b64: str = Field(
        ...,
        description="Clave privada de desencapsulación (dk) codificada en Base64.",
    )
    security_level: int = Field(
        ...,
        description="Nivel de seguridad configurado (512, 768 o 1024).",
        examples=[768],
    )


# ==============================================================================
# Esquemas: Cifrado Híbrido Autenticado (ML-KEM + AES-256-GCM)
# ==============================================================================

class HybridEncryptRequest(BaseModel):
    """Solicitud de cifrado híbrido poscuántico de una carga de datos."""

    ek_b64: str = Field(
        ...,
        description="Clave pública de encapsulación (ek) en Base64.",
    )
    plaintext: str = Field(
        ...,
        description="Texto en claro o carga Base64 a cifrar confidencialmente.",
        examples=["Mensaje confidencial post-cuántico"],
    )
    security_level: SecurityLevel = Field(
        default=768,
        description="Nivel de seguridad NIST ML-KEM.",
        examples=[768],
    )

    @field_validator("ek_b64")
    @classmethod
    def validate_ek(cls, v: str) -> str:
        return _validate_base64_field(v, "ek_b64")

    def get_ek_bytes(self) -> bytes:
        return base64.b64decode(self.ek_b64)


class HybridEncryptResponse(BaseModel):
    """Cápsula y carga cifrada mediante AES-256-GCM."""

    capsule_b64: str = Field(
        ...,
        description="Cápsula ML-KEM en Base64.",
    )
    nonce_b64: str = Field(
        ...,
        description="Nonce aleatorio de 12 bytes de AES-GCM en Base64.",
    )
    ciphertext_b64: str = Field(
        ...,
        description="Texto cifrado con tag de autenticación (16 bytes) en Base64.",
    )


class HybridDecryptRequest(BaseModel):
    """Solicitud de descifrado y verificación de integridad híbrida."""

    dk_b64: str = Field(
        ...,
        description="Clave privada de desencapsulación (dk) en Base64.",
    )
    capsule_b64: str = Field(
        ...,
        description="Cápsula ML-KEM recibida en Base64.",
    )
    nonce_b64: str = Field(
        ...,
        description="Nonce de 12 bytes de AES-GCM en Base64.",
    )
    ciphertext_b64: str = Field(
        ...,
        description="Texto cifrado con tag de autenticación en Base64.",
    )
    security_level: SecurityLevel = Field(
        default=768,
        description="Nivel de seguridad NIST ML-KEM.",
        examples=[768],
    )

    @field_validator("dk_b64")
    @classmethod
    def validate_dk(cls, v: str) -> str:
        return _validate_base64_field(v, "dk_b64")

    @field_validator("capsule_b64")
    @classmethod
    def validate_capsule(cls, v: str) -> str:
        return _validate_base64_field(v, "capsule_b64")

    @field_validator("nonce_b64")
    @classmethod
    def validate_nonce(cls, v: str) -> str:
        return _validate_base64_field(v, "nonce_b64")

    @field_validator("ciphertext_b64")
    @classmethod
    def validate_ciphertext(cls, v: str) -> str:
        return _validate_base64_field(v, "ciphertext_b64")

    def get_dk_bytes(self) -> bytes:
        return base64.b64decode(self.dk_b64)

    def get_capsule_bytes(self) -> bytes:
        return base64.b64decode(self.capsule_b64)

    def get_nonce_bytes(self) -> bytes:
        return base64.b64decode(self.nonce_b64)

    def get_ciphertext_bytes(self) -> bytes:
        return base64.b64decode(self.ciphertext_b64)


class HybridDecryptResponse(BaseModel):
    """Resultado del descifrado y verificación de autenticidad."""

    plaintext: str = Field(
        ...,
        description="Texto en claro recuperado íntegramente tras validar el tag AES-GCM.",
    )
