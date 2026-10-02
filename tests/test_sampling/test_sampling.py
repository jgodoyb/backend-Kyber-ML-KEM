import os
import pytest
from mlkem.constants import N, Q
from mlkem.sampling.sampling import sample_ntt, sample_poly_cbd


def test_sample_ntt_bounds_and_length():
    """Valida que SampleNTT devuelva 256 coeficientes en el rango [0, q-1]."""
    seed_34 = os.urandom(34)
    a_hat = sample_ntt(seed_34)

    assert len(a_hat) == N
    for coeff in a_hat:
        assert 0 <= coeff < Q


def test_sample_ntt_deterministic():
    """Valida que una misma semilla produzca idéntico polinomio NTT."""
    seed_34 = b"\x42" * 34
    res1 = sample_ntt(seed_34)
    res2 = sample_ntt(seed_34)
    assert res1 == res2


def test_sample_poly_cbd_bounds_eta2():
    """Para eta = 2, f[i] debe estar en {0, 1, 2} o {q-2, q-1} mod q."""
    seed = os.urandom(64 * 2)
    f = sample_poly_cbd(seed, eta=2)

    assert len(f) == N
    valid_values = {0, 1, 2, Q - 2, Q - 1}
    for coeff in f:
        assert coeff in valid_values


def test_sample_poly_cbd_bounds_eta3():
    """Para eta = 3, f[i] debe estar en {0, 1, 2, 3} o {q-3, q-2, q-1} mod q."""
    seed = os.urandom(64 * 3)
    f = sample_poly_cbd(seed, eta=3)

    assert len(f) == N
    valid_values = {0, 1, 2, 3, Q - 3, Q - 2, Q - 1}
    for coeff in f:
        assert coeff in valid_values


def test_sample_poly_cbd_deterministic():
    seed = b"\x13" * (64 * 2)
    res1 = sample_poly_cbd(seed, eta=2)
    res2 = sample_poly_cbd(seed, eta=2)
    assert res1 == res2


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sample_ntt(b"\x00" * 32)  # Faltan los 2 bytes de índices

    with pytest.raises(ValueError):
        sample_poly_cbd(b"\x00" * 100, eta=2)  # Longitud incorrecta

    with pytest.raises(ValueError):
        sample_poly_cbd(b"\x00" * 128, eta=4)  # eta inválido