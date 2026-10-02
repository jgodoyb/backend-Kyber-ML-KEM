import json
import os
import pytest
from mlkem.parameters.params import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from mlkem.core.internal import ml_kem_encaps_internal, ml_kem_decaps_internal

JSON_PATH = os.path.join(os.path.dirname(__file__), "encapdecap_acvp.json")

PARAM_MAP = {
    "ML-KEM-512": ML_KEM_512,
    "ML-KEM-768": ML_KEM_768,
    "ML-KEM-1024": ML_KEM_1024,
}


def load_acvp_encapdecap_cases():
    if not os.path.exists(JSON_PATH):
        return []

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    encap_cases = []
    decap_cases = []

    for group in data.get("testGroups", []):
        param_name = group.get("parameterSet")
        if param_name not in PARAM_MAP:
            continue
        params = PARAM_MAP[param_name]
        test_type = group.get("testType")  # AFT, etc.

        for test in group.get("tests", [])[:5]:
            tc_id = f"{param_name}-{test_type}-tcId-{test['tcId']}"

            # Si el caso contiene 'm', prueba Encapsulación
            if "m" in test and "ek" in test and "c" in test and "k" in test:
                encap_cases.append((
                    tc_id,
                    params,
                    test["ek"],
                    test["m"],
                    test["c"],
                    test["k"],
                ))

            # Si el caso contiene 'dk', 'c' y 'k', prueba Decapsulación
            if "dk" in test and "c" in test and "k" in test:
                decap_cases.append((
                    tc_id,
                    params,
                    test["dk"],
                    test["c"],
                    test["k"],
                ))

    return encap_cases, decap_cases


ENCAP_CASES, DECAP_CASES = load_acvp_encapdecap_cases()


@pytest.mark.skipif(not ENCAP_CASES, reason="Archivo encapdecap_acvp.json no encontrado")
@pytest.mark.parametrize("name, params, ek_hex, m_hex, expected_c, expected_k", ENCAP_CASES)
def test_nist_acvp_official_encaps(name, params, ek_hex, m_hex, expected_c, expected_k):
    """Verifica Encaps_internal contra salidas exactas del NIST."""
    ek = bytes.fromhex(ek_hex)
    m = bytes.fromhex(m_hex)

    k, c = ml_kem_encaps_internal(params, ek, m)

    assert c.hex() == expected_c.lower(), f"Ciphertext erróneo en {name}"
    assert k.hex() == expected_k.lower(), f"Clave K errónea en {name}"


@pytest.mark.skipif(not DECAP_CASES, reason="Archivo encapdecap_acvp.json no encontrado")
@pytest.mark.parametrize("name, params, dk_hex, c_hex, expected_k", DECAP_CASES)
def test_nist_acvp_official_decaps(name, params, dk_hex, c_hex, expected_k):
    """Verifica Decaps_internal (incluyendo rechazo implícito) contra el NIST."""
    dk = bytes.fromhex(dk_hex)
    c = bytes.fromhex(c_hex)

    k_recovered = ml_kem_decaps_internal(params, dk, c)

    assert k_recovered.hex() == expected_k.lower(), f"Clave K errónea al decapsular en {name}"