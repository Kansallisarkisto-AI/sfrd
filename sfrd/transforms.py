import numpy as np

from numba import njit


def lift_affine_to_fullres(M_small: np.ndarray, s_mov: float, s_ref: float) -> np.ndarray:
    """
    Convert an affine transformation computed on downscaled images to the coordinate space of the original full-scale images.

    M_small maps mov_small -> ref_small.
    mov_small = s_mov * mov_full coords
    ref_small = s_ref * ref_full coords

    => M_full maps mov_full -> ref_full:
       A_full = (s_mov/s_ref) * A_small
       t_full = (1/s_ref) * t_small

    Args:
        M_small: Affine transform in downscaled image coordionates.
        s_mov: Scale factor used for the image to be moved.
        s_ref: Scale factor used for the reference image.

    Returns:
        np.ndarray: Affine transform in full-resolution image coordinates.
    """
    M_full = M_small.astype(np.float64).copy()
    if s_ref <= 0:
        s_ref = 1.0
    if s_mov <= 0:
        s_mov = 1.0

    ratio = float(s_mov) / float(s_ref)
    M_full[:, 0:2] *= ratio
    # divide by s_ref to transform back to original 1.0 scale
    M_full[:, 2] /= float(s_ref)
    return M_full


def invert_affine(M: np.ndarray) -> np.ndarray | None:
    """
    Computes the inverse of a 2x3 affine transformation matrix.

    Args:
        M: Affine transformation matrix.

    Returns:
        np.ndarray: Inverted affine transformation matrix.

        Returns the identity transform if the matrix is singular or nearly
        singular.
    """
    A = M[:, :2]
    t = M[:, 2]
    det = float(np.linalg.det(A))
    if abs(det) < 1e-12:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float64)
    Ainv = np.linalg.inv(A)
    tinv = -Ainv @ t
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = Ainv
    out[:, 2] = tinv
    return out


def compose_affine(M2: np.ndarray, M1: np.ndarray) -> np.ndarray:
    """
    Return M = M2 @ M1 (apply M1 then M2).

    Args:
        M2: Second affine transform to apply.
        M1: First affine transform to apply.

    Returns:
        np.ndarray: Composed affine transform.
    """
    A1, t1 = M1[:, :2], M1[:, 2]
    A2, t2 = M2[:, :2], M2[:, 2]
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = A2 @ A1
    out[:, 2] = (A2 @ t1) + t2
    return out


def apply_affine(M: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    Applies an affine transform to 2D points.

    Args:
        M: 2x3 affine transformation matrix.
        xy: Array of shape `(N, 2)` containing 2D coordinates.

    Returns:
        np.ndarray: Transformed coordinates with shape `(N, 2)`.
    """
    return (xy @ M[:, :2].T) + M[:, 2]


@njit(cache=True, fastmath=True, nogil=True)
def invert_affine_numba(M):
    """Numba-accelerated affine transform inversion.

    Args:
        M: 2x3 affine transformation matrix.

    Returns:
        np.ndarray: Inverted affine transformation matrix.

        Returns the identity transform if the matrix is singular or nearly
        singular.
    """
    # Extract explicitly (no views)
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]

    tx = M[0, 2]
    ty = M[1, 2]

    det = a * d - b * c

    if abs(det) < 1e-12:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float64)

    inv_det = 1.0 / det

    # Inverse of 2x2
    a_inv = d * inv_det
    b_inv = -b * inv_det
    c_inv = -c * inv_det
    d_inv = a * inv_det

    # Inverse translation: -A_inv @ t
    t_inv_x = -(a_inv * tx + b_inv * ty)
    t_inv_y = -(c_inv * tx + d_inv * ty)

    out = np.empty((2, 3), dtype=np.float64)

    out[0, 0] = a_inv
    out[0, 1] = b_inv
    out[1, 0] = c_inv
    out[1, 1] = d_inv

    out[0, 2] = t_inv_x
    out[1, 2] = t_inv_y

    return out


@njit(cache=True, fastmath=True, nogil=True)
def compose_affine_numba(M2, M1):
    """
    Numba-accelerated affine transform composition.

    Computes:
        M = M2 @ M1

    Args:
        M2: Second affine transform to apply.
        M1: First affine transform to apply.

    Returns:
        np.ndarray: Composed affine transform.
    """
    # Extract everything explicitly
    a1 = M1[0, 0]
    b1 = M1[0, 1]
    c1 = M1[1, 0]
    d1 = M1[1, 1]
    tx1 = M1[0, 2]
    ty1 = M1[1, 2]

    a2 = M2[0, 0]
    b2 = M2[0, 1]
    c2 = M2[1, 0]
    d2 = M2[1, 1]
    tx2 = M2[0, 2]
    ty2 = M2[1, 2]

    out = np.empty((2, 3), dtype=np.float64)

    # A = A2 @ A1
    out[0, 0] = a2 * a1 + b2 * c1
    out[0, 1] = a2 * b1 + b2 * d1
    out[1, 0] = c2 * a1 + d2 * c1
    out[1, 1] = c2 * b1 + d2 * d1

    # t = A2 @ t1 + t2
    out[0, 2] = a2 * tx1 + b2 * ty1 + tx2
    out[1, 2] = c2 * tx1 + d2 * ty1 + ty2

    return out


@njit(cache=True, fastmath=True, nogil=True)
def apply_affine_numba(M, xy):
    """
    Numba-accelerated affine transform application.

    Args:
        M: 2x3 affine transformation matrix.
        xy: Array of shape `(N, 2)` containing 2D coordinates.

    Returns:
        np.ndarray: Transformed coordinates with shape `(N, 2)`.
    """
    n = xy.shape[0]
    out = np.empty((n, 2), dtype=np.float64)

    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    tx = M[0, 2]
    ty = M[1, 2]

    for i in range(n):
        x = xy[i, 0]
        y = xy[i, 1]

        out[i, 0] = a * x + b * y + tx
        out[i, 1] = c * x + d * y + ty

    return out
