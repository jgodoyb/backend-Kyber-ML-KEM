"""Q-Proof Kyber ML-KEM API.

Servicio REST de producción para criptografía post-cuántica
basado en ML-KEM (FIPS 203) y cifrado simétrico híbrido autenticado (AES-256-GCM).
"""

import os
import sys
import base64
from pathlib import Path
from typing import Dict

# Asegurar que el directorio raíz del repositorio esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cryptography.exceptions import InvalidTag

from mlkem.parameters.params import (
    MLKEMParameters,
    ML_KEM_512,
    ML_KEM_768,
    ML_KEM_1024,
)
from mlkem.crypto.hybrid import HybridPQC
from api.schemas import (
    KeygenRequest,
    KeygenResponse,
    HybridEncryptRequest,
    HybridEncryptResponse,
    HybridDecryptRequest,
    HybridDecryptResponse,
)

# Mapeo de niveles de seguridad aprobados por NIST FIPS 203
PARAMS_MAP: Dict[int, MLKEMParameters] = {
    512: ML_KEM_512,
    768: ML_KEM_768,
    1024: ML_KEM_1024,
}

# ==============================================================================
# Configuración de Rate Limiting (SlowAPI)
# ==============================================================================

RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
limiter = Limiter(key_func=get_remote_address, enabled=RATE_LIMIT_ENABLED)

# ==============================================================================
# Instanciación y Metadatos Formales de OpenAPI
# ==============================================================================

app = FastAPI(
    title="Q-Proof Kyber ML-KEM API",
    description="""
### API de Criptografía Post-Cuántica (PQC) - ML-KEM (FIPS 203)

Esta API proporciona primitivas criptográficas resistentes a ataques cuánticos mediante:
- **Generación de Claves (Keygen)**: Estandarizado por NIST en FIPS 203 (ML-KEM-512, ML-KEM-768 y ML-KEM-1024).
- **Cifrado Híbrido Autenticado**: Encapsulación poscuántica combinada con cifrado simétrico autenticado **AES-256-GCM** (AEAD).
- **Zero-Trust & Hardening**: Rate limiting mediante SlowAPI y validación estricta de integridad de datos.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==============================================================================
# Manejo Global de Excepciones Criptográficas
# ==============================================================================

@app.exception_handler(InvalidTag)
async def invalid_tag_exception_handler(request: Request, exc: InvalidTag):
    """Captura fallos de verificación de integridad en AES-GCM o manipulación de cápsula."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Decryption failed: integrity check failed"},
    )


@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    """Captura validaciones matemáticas y de formato en ML-KEM."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


# ==============================================================================
# Configuración Dinámica de CORS (Desarrollo y Producción)
# ==============================================================================

ENVIRONMENT = os.environ.get("ENVIRONMENT", "development").lower()
FRONTEND_URL = os.environ.get("FRONTEND_URL", "")

env_origins = [
    origin.strip().rstrip("/")
    for origin in FRONTEND_URL.split(",")
    if origin.strip()
]

if ENVIRONMENT == "production":
    allowed_origins = env_origins if env_origins else ["https://qproof.com"]
else:
    dev_defaults = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    # Eliminar duplicados preservando orden
    allowed_origins = list(dict.fromkeys(dev_defaults + env_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Endpoints: Estado y Diagnóstico
# ==============================================================================

@app.get("/", tags=["Diagnóstico & Estado"])
async def health_check():
    """Comprueba el estado operativo del microservicio criptográfico."""
    return {
        "status": "online",
        "service": "Q-Proof Kyber ML-KEM API",
        "version": "1.0.0",
        "standard": "NIST FIPS 203 (ML-KEM)",
        "supported_levels": [512, 768, 1024],
        "hybrid_cipher": "AES-256-GCM",
    }


# ==============================================================================
# Endpoints: Generación de Claves (Keygen)
# ==============================================================================

@app.post(
    "/keygen",
    response_model=KeygenResponse,
    summary="Generar par de claves ML-KEM",
    tags=["ML-KEM Key Exchange"],
)
@limiter.limit("10/minute")
async def keygen_endpoint(
    request: Request,
    payload: KeygenRequest = Body(default_factory=KeygenRequest),
) -> KeygenResponse:
    """Genera deterministamente con RBG un par de claves (ek, dk) según FIPS 203."""
    params = PARAMS_MAP.get(payload.security_level, ML_KEM_768)
    hybrid = HybridPQC(params)
    ek, dk = hybrid.keygen()

    return KeygenResponse(
        ek_b64=base64.b64encode(ek).decode("ascii"),
        dk_b64=base64.b64encode(dk).decode("ascii"),
        security_level=payload.security_level,
    )


# ==============================================================================
# Endpoints: Cifrado Híbrido Autenticado (ML-KEM + AES-256-GCM)
# ==============================================================================

@app.post(
    "/encrypt",
    response_model=HybridEncryptResponse,
    summary="Cifrado Híbrido Poscuántico",
    tags=["Cifrado Híbrido Autenticado"],
)
@limiter.limit("10/minute")
async def hybrid_encrypt_endpoint(
    request: Request,
    payload: HybridEncryptRequest,
) -> HybridEncryptResponse:
    """Cifra un mensaje mediante KEM post-cuántico y AES-256-GCM autenticado."""
    params = PARAMS_MAP.get(payload.security_level, ML_KEM_768)
    hybrid = HybridPQC(params)
    ek = payload.get_ek_bytes()

    # Convertir plaintext a bytes UTF-8
    plaintext_bytes = payload.plaintext.encode("utf-8")

    try:
        result = hybrid.encrypt_payload(ek, plaintext_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return HybridEncryptResponse(
        capsule_b64=base64.b64encode(result["capsule"]).decode("ascii"),
        nonce_b64=base64.b64encode(result["nonce"]).decode("ascii"),
        ciphertext_b64=base64.b64encode(result["ciphertext"]).decode("ascii"),
    )


@app.post(
    "/decrypt",
    response_model=HybridDecryptResponse,
    summary="Descifrado y Verificación Híbrida",
    tags=["Cifrado Híbrido Autenticado"],
)
@limiter.limit("10/minute")
async def hybrid_decrypt_endpoint(
    request: Request,
    payload: HybridDecryptRequest,
) -> HybridDecryptResponse:
    """Desencapsula la clave KEM y descifra/autentica el payload con AES-256-GCM."""
    params = PARAMS_MAP.get(payload.security_level, ML_KEM_768)
    hybrid = HybridPQC(params)
    dk = payload.get_dk_bytes()
    capsule = payload.get_capsule_bytes()
    nonce = payload.get_nonce_bytes()
    ciphertext = payload.get_ciphertext_bytes()

    try:
        plaintext_bytes = hybrid.decrypt_payload(
            dk=dk,
            capsule=capsule,
            nonce=nonce,
            ciphertext=ciphertext,
        )
    except InvalidTag:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Decryption failed: integrity check failed",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # Intentar decodificar como texto UTF-8; si es binario puro, codificar a Base64
    try:
        plaintext_str = plaintext_bytes.decode("utf-8")
    except UnicodeDecodeError:
        plaintext_str = base64.b64encode(plaintext_bytes).decode("ascii")

    return HybridDecryptResponse(plaintext=plaintext_str)


# ==============================================================================
# Punto de Entrada Principal (Ejecución Directa)
# ==============================================================================

if __name__ == "__main__":
    # Resolver de forma limpia la raíz del proyecto para importaciones absolutas
    root_path = Path(__file__).resolve().parent.parent
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
