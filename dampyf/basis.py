"""
    Local pseudomode basis construction.
"""

import numpy as np
import scipy.linalg as spla

from .structures import BasisData


def basis_initialization(pseudomodes):
    """
        Build the identity-plus-traceless operator basis in matrix and vectorized form.
    """
    basis = [
        [
            np.zeros((pseudomodes.fock_dim[n][q] ** 2, pseudomodes.fock_dim[n][q], pseudomodes.fock_dim[n][q]))
            for q in range(pseudomodes.Q[n])
        ]
        for n in range(pseudomodes.N)
    ]
    vecbasis = [
        [np.zeros((pseudomodes.fock_dim[n][q] ** 2, pseudomodes.fock_dim[n][q] ** 2)) for q in range(pseudomodes.Q[n])]
        for n in range(pseudomodes.N)
    ]

    fill_idtr_basis(pseudomodes, basis)
    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = pseudomodes.fock_dim[n][q] ** 2
            for i in range(dim):
                vecbasis[n][q][i] = basis[n][q][i].reshape(dim)
    return basis, vecbasis


def fill_idtr_basis(pseudomodes, basis):
    """
        Fill the identity-plus-traceless local operator basis.
    """
    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = pseudomodes.fock_dim[n][q]

            for i in range(dim - 1):
                basis[n][q][0, i, i] = 1.0 / np.sqrt(dim)
                basis[n][q][i + 1, i, i] = 1.0 / np.sqrt(2.0)
                basis[n][q][i + 1, i + 1, i + 1] = -1.0 / np.sqrt(2.0)
            basis[n][q][0, dim - 1, dim - 1] = 1.0 / np.sqrt(dim)

            ind = dim
            for i in range(dim):
                for j in range(i + 1, dim):
                    basis[n][q][ind, i, j] = 1.0 / np.sqrt(2.0)
                    basis[n][q][ind, j, i] = 1.0 / np.sqrt(2.0)
                    ind += 1
                    basis[n][q][ind, i, j] = 1.0 / np.sqrt(2.0)
                    basis[n][q][ind, j, i] = -1.0 / np.sqrt(2.0)
                    ind += 1

            orthonormalize_local_basis(pseudomodes, basis, n, q)

    return


def orthonormalize_local_basis(pseudomodes, basis, n, q):
    """
        Orthonormalize the pseudomode operator basis via QR decomposition.
    """
    dim = pseudomodes.fock_dim[n][q]
    basis_resh = np.zeros((dim * dim, dim * dim), dtype=float)

    for i in range(dim * dim):
        basis_resh[:, i] = basis[n][q][i].reshape(dim * dim)

    basis_q, _ = spla.qr(basis_resh)

    for i in range(dim * dim):
        basis[n][q][i] = basis_q[:, i].reshape(dim, dim)
        basis[n][q][i] /= -1.0

    expected = 1.0 / np.sqrt(dim)
    for i in range(dim):
        # Consistency check.
        if abs(basis[n][q][0, i, i] - expected) > 1.0e-10:
            raise ValueError("IdTr orthonormalization changed the identity component.")

    return


def fock_to_idtr(pseudomodes, idtr_basis):
    """
        Build the transformation from Fock basis to IdTr basis.
    """
    transform = [
        [np.zeros((pseudomodes.fock_dim[n][q] ** 2, pseudomodes.fock_dim[n][q] ** 2)) for q in range(pseudomodes.Q[n])]
        for n in range(pseudomodes.N)
    ]

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = pseudomodes.fock_dim[n][q]
            dim2 = dim**2
            fock_basis = np.zeros((dim2, dim, dim), dtype=float)

            ind = 0
            for i in range(dim):
                for j in range(dim):
                    fock_basis[ind, i, j] = 1.0
                    ind += 1

            for i in range(dim2):
                for j in range(dim2):
                    transform[n][q][i, j] = np.trace(np.conjugate(fock_basis[i].T) @ idtr_basis[n][q][j])

    return transform


def prepare_basis(pseudomodes):
    """
        Prepare all pseudomode basis objects used by DAMPF.
    """
    basis, vecbasis = basis_initialization(pseudomodes)
    transform = fock_to_idtr(pseudomodes, basis)

    return BasisData(basis=basis, vecbasis=vecbasis, fock_to_idtr=transform)
