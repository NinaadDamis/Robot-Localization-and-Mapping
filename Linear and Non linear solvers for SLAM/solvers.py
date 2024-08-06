'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular
from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    print("P_inv : A shape = {} , b shape = {}".format(A.shape,b.shape))
    b = b.reshape((-1,1))
    N = A.shape[1]
    x = np.zeros((N, ))
    x = inv(A.T @ A) @ (A.T @ b)
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    b = b.reshape((-1,1))
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    factorized = splu(csc_matrix(A.T@A), permc_spec='NATURAL')
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html -- > Get U matrix
    x = factorized.solve(A.T@b)
    U = factorized.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    b = b.reshape((-1,1))
    N = A.shape[1]
    x = np.zeros((N, ))
    U = eye(N)

    factorized = splu(A.T@A, permc_spec='COLAMD')
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html -- > Get U matrix
    x = factorized.solve(A.T@b)
    U = factorized.U
    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    # N = A.shape[1]
    # x = np.zeros((N, ))
    # R = eye(N)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve_triangular.html --> Add lower = false
    z , R , e , rank = rz(A, b, permc_spec='NATURAL')
    print("A shape = {} , b shape = {} ".format(A.shape, b.shape))
    print("SHapes = R = {} , z shape = {} , e shape = {} ".format(R.shape,z.shape,e.shape))
    x = spsolve_triangular(R,z,lower = False) # See ppt notes -> R is upper triangular. Get error if we dont put lower = false

    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    # N = A.shape[1]
    # x = np.zeros((N, ))
    # R = eye(N)
    z , R , e , rank = rz(A, b, permc_spec='COLAMD')
    print("e shape = ", e.shape)
    E = permutation_vector_to_matrix(e)
    print("E shape = ", E.shape)
    # R = R @ E.T
    # x = spsolve_triangular(R@E.T,z,lower = False) # See ppt notes -> R is upper triangular. Get error if we dont put lower = false
    x = spsolve_triangular(R,z,lower = False) # See ppt notes -> R is upper triangular. Get error if we dont put lower = false
    print("x shape = ", x.shape)
    x = E @ x # returns (2200,1)
    return x, R


def solve(A, b, method='default'):
    '''
    \param A (M, N) Jacobian matirx
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    '''
    M, N = A.shape

    fn_map = {
        'default': solve_default,
        'pinv': solve_pinv,
        'lu': solve_lu,
        'qr': solve_qr,
        'lu_colamd': solve_lu_colamd,
        'qr_colamd': solve_qr_colamd,
    }

    return fn_map[method](A, b)
