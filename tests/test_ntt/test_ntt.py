import os
import pytest
from mlkem.constants import N, Q, ZETA
from mlkem.ntt.ntt import (
    bit_rev_7, ZETAS, base_case_multiply, multiply_ntts, ntt, ntt_inv,
    poly_add, poly_sub, mat_vec_mul_ntt, mat_transpose_vec_mul_ntt, dot_product_ntt
)


def poly_mul_naive(f: list, g: list) -> list:
    """Multiplicación canónica en R_q = Z_q[X]/(X^256 + 1).

    Convolución directa con reducción X^256 = -1 mod q.
    """
    res = [0] * N
    for i in range(N):
        for j in range(N):
            deg = i + j
            term = (f[i] * g[j]) % Q
            if deg < N:
                res[deg] = (res[deg] + term) % Q
            else:
                # X^(256 + r) = -X^r
                res[deg - N] = (res[deg - N] - term) % Q
    return res


def test_zetas_table_consistency():
    """Valida la tabla precomputada ZETAS contra pow(ZETA, BitRev7(i), Q)."""
    assert len(ZETAS) == 128
    for i in range(128):
        expected = pow(ZETA, bit_rev_7(i), Q)
        assert ZETAS[i] == expected, f"Fallo en ZETAS[{i}]"


def test_ntt_invertibility_random_polys():
    """Prueba que NTT^-1(NTT(f)) == f para múltiples polinomios aleatorios."""
    for _ in range(10):
        f = [int.from_bytes(os.urandom(2), "little") % Q for _ in range(N)]
        f_hat = ntt(f)
        recovered = ntt_inv(f_hat)
        assert recovered == f


def test_ntt_ring_multiplication_homomorphism():
    """Ecuación 4.9 del FIPS 203:

    f * g en R_q == NTT^-1(NTT(f) *_T_q NTT(g))
    Valida la congruencia matemática entre la multiplicación lenta canónica
    y la multiplicación en el dominio NTT.
    """
    for _ in range(5):
        f = [int.from_bytes(os.urandom(2), "little") % Q for _ in range(N)]
        g = [int.from_bytes(os.urandom(2), "little") % Q for _ in range(N)]

        expected_prod = poly_mul_naive(f, g)

        f_hat = ntt(f)
        g_hat = ntt(g)
        prod_hat = multiply_ntts(f_hat, g_hat)
        ntt_prod = ntt_inv(prod_hat)

        assert ntt_prod == expected_prod


def test_base_case_multiply():
    """Valida la multiplicación en el cuerpo cuadrático mod X^2 - gamma."""
    # (2 + 3X) * (4 + 5X) = 8 + 10X + 12X + 15X^2 = (8 + 15*gamma) + 22X
    gamma = 17
    a0, a1 = 2, 3
    b0, b1 = 4, 5
    c0, c1 = base_case_multiply(a0, a1, b0, b1, gamma)
    assert c0 == (2 * 4 + 3 * 5 * 17) % Q
    assert c1 == (2 * 5 + 3 * 4) % Q


def test_linear_algebra_ntt():
    """Prueba que las operaciones matriciales y de producto escalar

    cumplan la asociatividad básica.
    """
    k = 3
    zero_poly = [0] * N
    one_poly = [1] + [0] * 255  # Polinomio '1'

    A_hat = [[[0] * N for _ in range(k)] for _ in range(k)]
    s_hat = [[0] * N for _ in range(k)]

    # Matriz identidad
    for i in range(k):
        A_hat[i][i] = ntt(one_poly)
        s_hat[i] = ntt([i + 1] + [0] * 255)

    res = mat_vec_mul_ntt(A_hat, s_hat)
    for i in range(k):
        assert res[i] == s_hat[i]

    # Matriz traspuesta
    res_t = mat_transpose_vec_mul_ntt(A_hat, s_hat)
    assert res_t == res


def test_invalid_lengths():
    with pytest.raises(ValueError):
        ntt([0] * 128)

    with pytest.raises(ValueError):
        ntt_inv([0] * 300)

    with pytest.raises(ValueError):
        multiply_ntts([0] * 256, [0] * 128)