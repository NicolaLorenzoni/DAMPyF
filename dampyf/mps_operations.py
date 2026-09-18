"""
    DAMPF-specific matrix-product-object library.
"""

from collections.abc import Iterable, Sequence
from os import PathLike

import numpy as np
from numpy.linalg import qr
from scipy import linalg


class MatrixProductObject:
    """
        Matrix product object class
    """

    def __init__(self, local_tensors, canonical_form=None, copy=True):
        self._local_tensors = []
        for tensor in local_tensors:
            self._local_tensors.append(np.array(tensor, copy=copy))

        self._check_local_tensors()

        if canonical_form is None:
            self.left_canonical = 0
            self.right_canonical = len(self._local_tensors)
        else:
            self.left_canonical = int(canonical_form[0])
            self.right_canonical = int(canonical_form[1])
            self._check_canonical_form()

    def _check_local_tensors(self):
        if len(self._local_tensors) == 0:
            raise ValueError("At least one local tensor is required.")

        for tensor in self._local_tensors:
            if tensor.ndim < 2:
                raise ValueError("Each local tensor must have at least two virtual dimensions.")

        if self._local_tensors[0].shape[0] != 1:
            raise ValueError("The left boundary rank must be 1.")
        if self._local_tensors[-1].shape[-1] != 1:
            raise ValueError("The right boundary rank must be 1.")

        for left_tensor, right_tensor in zip(self._local_tensors[:-1], self._local_tensors[1:]):
            if left_tensor.shape[-1] != right_tensor.shape[0]:
                raise ValueError("Neighbouring local tensors have incompatible virtual ranks.")

    def _check_canonical_form(self):
        if not (0 <= self.left_canonical < len(self._local_tensors)):
            raise ValueError("Invalid left canonical-form boundary.")
        if not (0 < self.right_canonical <= len(self._local_tensors)):
            raise ValueError("Invalid right canonical-form boundary.")
        if self.left_canonical >= self.right_canonical:
            raise ValueError("Invalid canonical-form interval.")

    def copy(self):
        """
            Return a deep copy.
        """
        return type(self)(self._local_tensors, canonical_form=self.canonical_form, copy=True)

    def __len__(self):
        return len(self._local_tensors)

    @property
    def local_tensors(self):
        """
            Direct access to the local tensors.
        """
        return self._local_tensors

    @property
    def size(self):
        """
            Number of scalar tensor entries stored by this object.
        """
        return sum(tensor.size for tensor in self._local_tensors)

    @property
    def dtype(self):
        """
            Common NumPy dtype of all local tensors.
        """
        return np.result_type(*[tensor.dtype for tensor in self._local_tensors])

    @property
    def bond_dimensions(self):
        """
            Internal virtual bond dimensions.
        """
        return tuple(tensor.shape[0] for tensor in self._local_tensors[1:])

    @property
    def local_physical_dimensions(self):
        """
            Local physical dimensions at each site.
        """
        return tuple(tensor.shape[1:-1] for tensor in self._local_tensors)

    @property
    def local_physical_legs(self):
        """
            Number of local physical tensor legs at each site.
        """
        return tuple(tensor.ndim - 2 for tensor in self._local_tensors)

    @property
    def canonical_form(self):
        return self.left_canonical, self.right_canonical

    def save(self, target):
        """
            Save this matrix-product object to an HDF5 file or group.
        """
        if isinstance(target, (str, PathLike)):
            import h5py

            with h5py.File(target, "w") as output_file:
                self.save(output_file)
            return

        target.attrs["num_sites"] = len(self)
        target.attrs["left_canonical"] = self.left_canonical
        target.attrs["right_canonical"] = self.right_canonical

        for site, tensor in enumerate(self._local_tensors):
            target[str(site)] = tensor

        return

    @classmethod
    def load(cls, source):
        """
            Load a matrix-product object from an HDF5 file or group.
        """
        if isinstance(source, (str, PathLike)):
            import h5py

            with h5py.File(source, "r") as input_file:
                return cls.load(input_file)

        num_sites = int(source.attrs["num_sites"])

        local_tensors = []
        for site in range(num_sites):
            local_tensors.append(source[str(site)][()])

        canonical_form = (int(source.attrs["left_canonical"]), int(source.attrs["right_canonical"]))

        return cls(local_tensors, canonical_form=canonical_form, copy=False)

    def conj(self):
        """
            Return the complex conjugate.
        """
        local_tensors = []
        for tensor in self._local_tensors:
            local_tensors.append(tensor.conj())
        return type(self)(local_tensors, canonical_form=self.canonical_form, copy=False)

    def __add__(self, summand):
        """
            Return the matrix-product sum using block-diagonal virtual bonds.
        """
        if len(self) != len(summand):
            raise ValueError(f"Length mismatch in matrix-product addition: {len(self)} != {len(summand)}")
        if self.local_physical_dimensions != summand.local_physical_dimensions:
            raise ValueError("Physical dimensions must agree in matrix-product addition.")

        if len(self) == 1:
            return type(self)([self._local_tensors[0] + summand.local_tensors[0]], copy=False)

        local_tensors = []
        local_tensors.append(np.concatenate((self._local_tensors[0], summand.local_tensors[0]), axis=-1))

        for left_tensor, right_tensor in zip(self._local_tensors[1:-1], summand.local_tensors[1:-1]):
            local_tensors.append(_local_add([left_tensor, right_tensor]))

        local_tensors.append(np.concatenate((self._local_tensors[-1], summand.local_tensors[-1]), axis=0))
        return type(self)(local_tensors, copy=False)

    def __sub__(self, subtrahend):
        return self + (-1) * subtrahend

    def __mul__(self, factor):
        """
            Return scalar multiplication by ``factor``.
        """
        if not np.isscalar(factor):
            raise NotImplementedError("Only scalar multiplication is supported.")

        local_tensors = []
        for tensor in self._local_tensors:
            local_tensors.append(tensor.copy())
        local_tensors[self.left_canonical] = factor * local_tensors[self.left_canonical]
        return type(self)(local_tensors, canonical_form=self.canonical_form, copy=False)

    def __imul__(self, factor):
        """
            In-place scalar multiplication by ``factor``.
        """
        if not np.isscalar(factor):
            raise NotImplementedError("Only scalar multiplication is supported.")

        self._local_tensors[self.left_canonical] = factor * self._local_tensors[self.left_canonical]
        return self

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def __truediv__(self, divisor):
        if not np.isscalar(divisor):
            raise NotImplementedError("Only scalar division is supported.")
        return self.__mul__(1 / divisor)

    def __itruediv__(self, divisor):
        if not np.isscalar(divisor):
            raise NotImplementedError("Only scalar division is supported.")
        return self.__imul__(1 / divisor)

    def canonicalize(self):
        """
            Move the orthogonality centre to one boundary, choosing the shorter path.
        """
        if len(self) == 1:
            return

        if self.left_canonical < len(self) - self.right_canonical:
            self._canonicalize_right_to_left(1)
        else:
            self._canonicalize_left_to_right(len(self) - 1)

        return

    def _canonicalize_left_to_right(self, target_left_boundary):
        """
            QR sweep from left to right until ``left_canonical = target_left_boundary``.
        """
        if not 0 <= target_left_boundary < len(self):
            raise IndexError(f"Invalid left-canonical boundary {target_left_boundary}.")

        for site in range(self.left_canonical, target_left_boundary):
            tensor = self._local_tensors[site]
            q_matrix, r_matrix = qr(tensor.reshape((-1, tensor.shape[-1])))
            self._local_tensors[site] = q_matrix.reshape(tensor.shape[:-1] + (-1,))
            self._local_tensors[site + 1] = _matrix_dot(r_matrix, self._local_tensors[site + 1])
            self.left_canonical = site + 1
            self.right_canonical = max(self.right_canonical, site + 2)

        return

    def _canonicalize_right_to_left(self, target_right_boundary):
        """
            QR sweep from right to left until ``right_canonical = target_right_boundary``.
        """
        if not 0 < target_right_boundary <= len(self):
            raise IndexError(f"Invalid right-canonical boundary {target_right_boundary}.")

        for site in range(self.right_canonical - 1, target_right_boundary - 1, -1):
            tensor = self._local_tensors[site]
            q_matrix, r_matrix = qr(tensor.reshape((tensor.shape[0], -1)).T)
            self._local_tensors[site - 1] = _matrix_dot(self._local_tensors[site - 1], r_matrix.T)
            self._local_tensors[site] = q_matrix.T.reshape((-1,) + tensor.shape[1:])
            self.left_canonical = min(self.left_canonical, site - 1)
            self.right_canonical = site

        return

    def compress(self, rank=None, relerr=None, direction=None, canonicalize=True, svd_method="standard"):
        """
            Compress this object in place using the selected SVD backend.
        """
        if len(self) == 1:
            return

        if direction is None:
            if len(self) - self.right_canonical > self.left_canonical:
                direction = "left"
            else:
                direction = "right"

        if direction == "right":
            if canonicalize:
                self._canonicalize_right_to_left(1)
            self._compress_left_to_right(rank, relerr, svd_method)
            return

        if direction == "left":
            if canonicalize:
                self._canonicalize_left_to_right(len(self) - 1)
            self._compress_right_to_left(rank, relerr, svd_method)
            return

        raise ValueError(f"Invalid SVD compression direction: {direction}")

    def _choose_rank(self, singular_values, maximum_allowed_rank, requested_rank, relerr):
        if requested_rank is None:
            requested_rank = maximum_allowed_rank
        requested_rank = int(requested_rank)
        if requested_rank <= 0:
            raise ValueError(f"Cannot compress to rank {requested_rank}.")

        if relerr is None:
            return min(requested_rank, len(singular_values), maximum_allowed_rank)

        if not (0.0 <= relerr <= 1.0):
            raise ValueError(f"Invalid relative compression error {relerr}.")

        if len(singular_values) == 0 or singular_values[0] < 1.0e-15:
            tolerance_rank = 1
        else:
            cumulative = np.cumsum(singular_values) / np.sum(singular_values)
            tolerance_rank = int(np.searchsorted(cumulative, 1.0 - relerr) + 1)

        return min(requested_rank, tolerance_rank, len(singular_values), maximum_allowed_rank)

    def _compress_left_to_right(self, rank, relerr, svd_method):
        for site in range(len(self) - 1):
            tensor = self._local_tensors[site]
            matrix_shape = (-1, tensor.shape[-1])

            if relerr is None:
                svd_rank = rank
            else:
                svd_rank = None

            u_matrix, singular_values, vh_matrix = _compute_svd(tensor.reshape(matrix_shape), svd_rank, svd_method)
            kept_rank = self._choose_rank(singular_values, tensor.shape[-1], rank, relerr)
            self._local_tensors[site] = u_matrix[:, :kept_rank].reshape(tensor.shape[:-1] + (kept_rank,))
            self._local_tensors[site + 1] = _matrix_dot(
                singular_values[:kept_rank, None] * vh_matrix[:kept_rank, :], self._local_tensors[site + 1]
            )
            self.left_canonical = site + 1
            self.right_canonical = max(self.right_canonical, site + 2)

        return

    def _compress_right_to_left(self, rank, relerr, svd_method):
        for site in range(len(self) - 1, 0, -1):
            tensor = self._local_tensors[site]
            matrix_shape = (tensor.shape[0], -1)

            if relerr is None:
                svd_rank = rank
            else:
                svd_rank = None

            u_matrix, singular_values, vh_matrix = _compute_svd(tensor.reshape(matrix_shape), svd_rank, svd_method)
            kept_rank = self._choose_rank(singular_values, tensor.shape[0], rank, relerr)
            self._local_tensors[site - 1] = _matrix_dot(
                self._local_tensors[site - 1], u_matrix[:, :kept_rank] * singular_values[None, :kept_rank]
            )
            self._local_tensors[site] = vh_matrix[:kept_rank, :].reshape((kept_rank,) + tensor.shape[1:])
            self.left_canonical = min(self.left_canonical, site - 1)
            self.right_canonical = site

        return


def mp_dot(mpo_or_mpa, mps_or_mpa, axes=(-1, 0), astype=None):
    """
        Contract two matrix-product objects site by site.
    """
    if len(mpo_or_mpa) != len(mps_or_mpa):
        raise ValueError(f"Length mismatch in mp_dot: {len(mpo_or_mpa)} != {len(mps_or_mpa)}")

    contraction_axes = _shift_physical_axes(axes)
    local_tensors = []
    for left_tensor, right_tensor in zip(mpo_or_mpa.local_tensors, mps_or_mpa.local_tensors):
        local_tensors.append(_local_dot(left_tensor, right_tensor, contraction_axes))

    if astype is None:
        astype = type(mpo_or_mpa)

    return astype(local_tensors, copy=False)


def apply_mpo(mpo, mps_or_mpo):
    """
        Apply an MPO-like object to an MPS- or MPO-like object.
    """
    return mp_dot(mpo, mps_or_mpo, axes=(-1, 0))


def norm(mpo_or_mps):
    """
        Frobenius/Hilbert-space norm of a matrix-product object.
    """
    mpo_or_mps.canonicalize()
    left_canonical, right_canonical = mpo_or_mps.canonical_form

    if right_canonical == 1:
        return np.linalg.norm(mpo_or_mps.local_tensors[0])
    if left_canonical == len(mpo_or_mps) - 1:
        return np.linalg.norm(mpo_or_mps.local_tensors[-1])

    raise ValueError("Canonicalization error while computing matrix-product-object norm.")


def maximum_bond_dimensions(local_dimensions):
    """
        Maximal open-boundary bond dimensions for the given local physical dimensions.
    """
    flattened_dimensions = np.array(
        [int(np.prod(local_dimension)) for local_dimension in local_dimensions], dtype=object
    )
    bond_dimensions = []
    for cut in range(1, len(flattened_dimensions)):
        bond_dimensions.append(min(np.prod(flattened_dimensions[:cut]), np.prod(flattened_dimensions[cut:])))
    return bond_dimensions


def _shift_physical_axes(axes):
    if isinstance(axes[0], Sequence) and not isinstance(axes[0], int):
        shifted_axes = []
        for axes_part in axes:
            shifted_axes_part = []
            for axis in axes_part:
                if axis >= 0:
                    shifted_axes_part.append(axis + 1)
                else:
                    shifted_axes_part.append(axis - 1)
            shifted_axes.append(tuple(shifted_axes_part))
        return tuple(shifted_axes)

    shifted_axes = []
    for axis in axes:
        if axis >= 0:
            shifted_axes.append(axis + 1)
        else:
            shifted_axes.append(axis - 1)
    return tuple(shifted_axes)


def _local_dot(left_tensor, right_tensor, axes):
    """
        Local tensor contraction underlying ``mp_dot``.
    """
    if isinstance(axes[0], Sequence) and not isinstance(axes[0], int):
        contracted_legs_left = len(axes[0])
        contracted_legs_right = len(axes[1])
    else:
        contracted_legs_left = 1
        contracted_legs_right = 1

    if contracted_legs_left != contracted_legs_right:
        raise ValueError("The number of contracted legs must agree.")

    result = np.tensordot(left_tensor, right_tensor, axes=axes)
    result = np.rollaxis(result, left_tensor.ndim - contracted_legs_left, 1)
    destination = left_tensor.ndim + right_tensor.ndim - contracted_legs_left - contracted_legs_right - 1
    result = np.rollaxis(result, left_tensor.ndim - contracted_legs_left, destination)

    return result.reshape(
        (left_tensor.shape[0] * right_tensor.shape[0],)
        + result.shape[2:-2]
        + (left_tensor.shape[-1] * right_tensor.shape[-1],)
    )


def _local_add(local_tensors):
    """
        Block-diagonal local tensor sum for non-boundary sites.
    """
    physical_shape = local_tensors[0].shape[1:-1]
    result_shape = (
        (sum(tensor.shape[0] for tensor in local_tensors),)
        + physical_shape
        + (sum(tensor.shape[-1] for tensor in local_tensors),)
    )
    result = np.zeros(result_shape, dtype=np.result_type(*[tensor.dtype for tensor in local_tensors]))

    left_position = 0
    right_position = 0
    for tensor in local_tensors:
        new_left_position = left_position + tensor.shape[0]
        new_right_position = right_position + tensor.shape[-1]
        result[left_position:new_left_position, ..., right_position:new_right_position] = tensor
        left_position = new_left_position
        right_position = new_right_position

    return result


def _matrix_dot(left, right, axes=((-1,), (0,))):
    return np.tensordot(left, right, axes=axes)


def _compute_svd(matrix, rank, svd_method):
    if svd_method == "standard":
        if rank is None:
            return _full_svd(matrix)
        return _truncated_svd(matrix, rank)
    if svd_method == "qr":
        return _qr_svd(matrix, rank)
    if svd_method == "eig":
        return _eig_svd(matrix, rank)
    raise ValueError("svd_method must be 'standard', 'qr', or 'eig'.")


def _truncate_svd_result(u_matrix, singular_values, vh_matrix, rank):
    if rank is None:
        kept_rank = len(singular_values)
    else:
        kept_rank = min(int(rank), len(singular_values))
        if kept_rank <= 0:
            raise ValueError(f"Cannot compute an SVD with rank {rank}.")

    return u_matrix[:, :kept_rank], singular_values[:kept_rank], vh_matrix[:kept_rank, :]


def _truncated_svd(matrix, rank):
    u_matrix, singular_values, vh_matrix = linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")
    return _truncate_svd_result(u_matrix, singular_values, vh_matrix, rank)


def _full_svd(matrix):
    return linalg.svd(matrix, full_matrices=False, lapack_driver="gesvd")


def _qr_svd(matrix, rank):
    matrix = np.asarray(matrix)
    num_rows, num_cols = matrix.shape

    if num_rows >= num_cols:
        q_matrix, r_matrix = linalg.qr(matrix, mode="economic")
        u_small, singular_values, vh_matrix = linalg.svd(r_matrix, full_matrices=False, lapack_driver="gesvd")
        u_matrix = q_matrix @ u_small
    else:
        q_matrix, r_matrix = linalg.qr(matrix.conj().T, mode="economic")
        u_matrix, singular_values, vh_small = linalg.svd(r_matrix.conj().T, full_matrices=False, lapack_driver="gesvd")
        vh_matrix = vh_small @ q_matrix.conj().T

    return _truncate_svd_result(u_matrix, singular_values, vh_matrix, rank)


def _eig_svd(matrix, rank, cutoff=1.0e-14):
    matrix = np.asarray(matrix)
    num_rows, num_cols = matrix.shape

    if num_rows <= num_cols:
        gram_matrix = matrix @ matrix.conj().T
        eigenvalues, u_matrix = linalg.eigh(gram_matrix)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        u_matrix = u_matrix[:, order]

        singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
        if rank is None:
            kept_rank = len(singular_values)
        else:
            kept_rank = min(int(rank), len(singular_values))
            if kept_rank <= 0:
                raise ValueError(f"Cannot compute an SVD with rank {rank}.")

        u_matrix = u_matrix[:, :kept_rank]
        singular_values = singular_values[:kept_rank]
        safe_singular_values = np.where(singular_values > cutoff, singular_values, cutoff)
        vh_matrix = (u_matrix.conj().T @ matrix) / safe_singular_values[:, None]

    else:
        gram_matrix = matrix.conj().T @ matrix
        eigenvalues, v_matrix = linalg.eigh(gram_matrix)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        v_matrix = v_matrix[:, order]

        singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))
        if rank is None:
            kept_rank = len(singular_values)
        else:
            kept_rank = min(int(rank), len(singular_values))
            if kept_rank <= 0:
                raise ValueError(f"Cannot compute an SVD with rank {rank}.")

        v_matrix = v_matrix[:, :kept_rank]
        singular_values = singular_values[:kept_rank]
        safe_singular_values = np.where(singular_values > cutoff, singular_values, cutoff)
        u_matrix = (matrix @ v_matrix) / safe_singular_values[None, :]
        vh_matrix = v_matrix.conj().T

    return u_matrix, singular_values, vh_matrix


def _normalize_local_dimensions(sites, local_dimensions):
    """
        Convert the accepted ``ldim`` formats to one tuple per site.
    """
    if isinstance(local_dimensions, Iterable) and not isinstance(local_dimensions, (str, bytes)):
        local_dimensions = tuple(local_dimensions)
    else:
        local_dimensions = (local_dimensions,)

    if len(local_dimensions) == 0:
        raise ValueError("At least one local dimension is required.")

    first_entry = local_dimensions[0]
    if not isinstance(first_entry, Iterable) or isinstance(first_entry, (str, bytes)):
        local_dimensions = (local_dimensions,) * sites

    if len(local_dimensions) != sites:
        raise ValueError("The number of local dimensions must match the number of sites.")

    return tuple(tuple(int(dim) for dim in dimensions) for dimensions in local_dimensions)


def _normalize_ranks(sites, rank, local_dimensions, force_rank):
    """
        Convert a scalar or iterable rank specification to internal ranks.
    """
    if isinstance(rank, Iterable) and not isinstance(rank, (str, bytes)):
        ranks = tuple(int(value) for value in rank)
    else:
        ranks = (int(rank),) * (sites - 1)

    if len(ranks) != sites - 1:
        raise ValueError("Rank specification must have length sites - 1.")

    if not force_rank:
        ranks = tuple(
            min(requested_rank, maximal_rank)
            for requested_rank, maximal_rank in zip(ranks, maximum_bond_dimensions(local_dimensions))
        )

    return ranks


def create_zero_matrix_product_object(sites, ldim, rank, force_rank=False):
    """
        Return a zero matrix-product object with prescribed local dimensions.
    """
    sites = int(sites)
    if sites <= 0:
        raise ValueError("The number of sites must be positive.")

    local_dimensions = _normalize_local_dimensions(sites, ldim)
    ranks = _normalize_ranks(sites, rank, local_dimensions, force_rank)
    boundary_ranks = (1,) + ranks + (1,)

    local_tensors = []
    for site, dimensions in enumerate(local_dimensions):
        tensor_shape = (boundary_ranks[site],) + tuple(dimensions) + (boundary_ranks[site + 1],)
        local_tensors.append(np.zeros(tensor_shape))

    return MatrixProductObject(local_tensors, copy=False)
