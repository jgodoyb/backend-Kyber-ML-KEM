import json
import os
import pytest
from mlkem.parameters.params import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from mlkem.core.internal import ml_kem_keygen_internal

JSON_PATH = os.path.join(os.path.dirname(__file__), "keygen_acvp.json")

PARAM_MAP = {
    "ML-KEM-512": ML_KEM_512,
    "ML-KEM-768": ML_KEM_768,
    "ML-KEM-1024": ML_KEM_1024,
}


def load_acvp_keygen_cases():
    if not os.path.exists(JSON_PATH):
        return []

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = []
    # La estructura ACVP organiza los casos por testGroups
    for group in data.get("testGroups", []):
        param_name = group.get("parameterSet")
        if param_name not in PARAM_MAP:
            continue
        params = PARAM_MAP[param_name]

        # Tomamos los primeros 5 casos de cada grupo para verificar exhaustivamente
        for test in group.get("tests", [])[:5]:
            cases.append((
                f"{param_name}-tcId-{test['tcId']}",
                params,
                test["d"],
                test["z"],
                test["ek"],
                test["dk"],
            ))
    return cases


TEST_CASES = load_acvp_keygen_cases()


@pytest.mark.skipif(not TEST_CASES, reason="Archivo keygen_acvp.json no encontrado")
@pytest.mark.parametrize("name, params, d_hex, z_hex, expected_ek, expected_dk", TEST_CASES)
def test_nist_acvp_official_keygen(name, params, d_hex, z_hex, expected_ek, expected_dk):
    """Contrasta la generación determinista contra las salidas exactas del NIST ACVP."""
    d = bytes.fromhex(d_hex)
    z = bytes.fromhex(z_hex)

    ek, dk = ml_kem_keygen_internal(params, d, z)

    assert ek.hex() == expected_ek.lower(), f"Discrepancia en ek en el caso {name}"
    assert dk.hex() == expected_dk.lower(), f"Discrepancia en dk en el caso {name}"