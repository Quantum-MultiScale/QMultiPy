import warnings
from functools import partial

import numpy as np
from scipy import ndimage, signal

from qmultipy.grid import DirectGrid, ReciprocalGrid


def direct_to_otherdirect(field, othergrid):
    """
    Downsample a direct field onto a coarser direct grid by FFT truncation.

    Assumes both grids share the same lattice. The target grid must have
    fewer (or equal) points along each axis.
    """
    if hasattr(othergrid, "grid"):
        othergrid = othergrid.grid
    if not isinstance(othergrid, DirectGrid):
        raise TypeError("othergrid must be a DirectGrid or DirectField")

    if not np.allclose(field.grid.lattice, othergrid.lattice):
        raise ValueError("direct_to_otherdirect requires matching lattices")

    if np.any(othergrid.nrR > field.grid.nrR):
        raise ValueError("othergrid must be coarser than field.grid")

    if not field.grid.full or not othergrid.full:
        raise NotImplementedError(
            "direct_to_otherdirect currently requires full grids"
        )

    rank = field.rank
    axes = (-3, -2, -1)

    recip = field.fft()
    fft_data = np.fft.fftshift(np.asarray(recip), axes=axes)

    slices = []
    for dim_old, dim_new in zip(field.grid.nrR, othergrid.nrR):
        start = (dim_old - dim_new) // 2
        end = start + dim_new
        slices.append(slice(start, end))

    if rank > 1:
        slices = (slice(None), *slices)
    else:
        slices = tuple(slices)

    fft_data = fft_data[slices]
    fft_data = np.fft.ifftshift(fft_data, axes=axes)

    recip_coarse = recip.__class__(
        grid=othergrid.get_reciprocal(),
        data=fft_data,
        rank=rank,
        cplx=field.cplx,
    )
    return recip_coarse.ifft(check_real=not field.cplx, force_real=not field.cplx)


def direct_to_GTOs_full(field, mol, grid_level=4):
    from pyscf.dft.gen_grid import Grids

    # f(r) = sum_mu_nu coeffs_mu_nu * phi_mu(r) * phi_nu(r) where phi_mu(r) is the GTO
    '''
    Function returns the coefficients gamma_mu_nu of the expansion of f(r) in terms of the GTOs.
    Args:
        field: QMultiPy direct field object
        mol: PySCF mol class
    Returns:
        coeffs: numpy array of size (nao,nao)
    '''
    # get the electrostatic potetnial from direct field
    fg = field.fft()
    fg *= 4 * np.pi * fg.grid.invgg
    v_H = fg.ifft(force_real=True)
    # Get the atomic grid
    grids = Grids(mol)
    grids.level = grid_level
    grids.build()
    # put the field on the atomic grid
    field_atomic = direct_to_atomic_accurate(v_H, grids.coords)
    # get mat_mu nu = <mu nu | 1/r12 | field >
    mat = direct_potential_atomic_to_ao(mol, field_atomic, grids.coords, grids.weights)
    # set up linear system for DF problem: mat = eri * c
    eri = mol.intor("int2e", aosym="s1")
    n = mat.shape[0]
    N = n**2
    # Use pseudoinverse since eri matrix may be singular
    # coeffs = np.linalg.pinv(eri.transpose(0,2,1,3).reshape(N,N)) @ mat.flatten()
    coeffs = np.linalg.pinv(eri.reshape(N, N)) @ mat.flatten()

    # check if field is well represented
    ovlp = mol.intor("int1e_ovlp").flatten()
    int_field = field.integral()
    int_mat = np.einsum("m,m->", coeffs, ovlp)

    if int_mat - int_field > 1e-6:
        print("int_mat", int_mat)
        print("int_field", int_field)
        warnings.warn("Field is not well represented by GTOs")
    return coeffs.reshape(n, n)


def direct_to_GTOs(field, mol, grid_level=4):
    from pyscf.dft.gen_grid import Grids

    # f(r) - f_0(r) = sum_mu c_mu * phi_mu(r) where phi_mu(r) is the GTO
    # f_0(r) is some positive definite function (i.e., SAD)
    '''
    Function returns the coefficients c_mu of the expansion of f(r) in terms of the GTOs.
    Args:
        field: QMultiPy direct field object
        mol: PySCF mol class
    Returns:
        c: numpy array of size (nao)
    '''

    # Get the atomic grid
    grids = Grids(mol)
    grids.level = grid_level
    grids.build()

    # put the field on the atomic grid
    field_atomic = direct_to_atomic_accurate(field, grids.coords)

    # put direct_atomic to AOs
    c = direct_atomic_to_ao(mol, field_atomic, grids.coords, grids.weights)

    # check if field is well represented
    # <mu| field> = sum_nu c_nu * <mu|nu>

    ovlp = mol.intor("int1e_ovlp")
    c_field = np.einsum("mn,mn->", c, ovlp)
    if np.any(np.abs(c - c_field) > 1e-6):
        warnings.warn("Field is not well represented by GTOs")
    return c


def direct_atomic_to_ao(mol, field_atomic, grid_atomic_points, grid_atomic_weights):
    '''
    Function returns the <field_atomic|mu> coeffs where field_atomic is a function on atomic grid.

    Args:
        mol: PySCF mol class
        field_atomic: direct field-like object given as numpy.array on an atomic grid
        grid_atomic_points: numpy.array of size (npoints,3)
        grid_atomic_weights: numpy.array of size (npoints)

    Returns:
        coeffs: numpy.array of size (nao,nao)
    '''

    from pyscf.dft.gen_grid import BLKSIZE
    from pyscf.dft.numint import _dot_ao_ao, _scale_ao, eval_ao

    ao = eval_ao(mol, grid_atomic_points, deriv=0)
    ngrids, nao = ao.shape
    non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    aow = _scale_ao(ao, grid_atomic_weights * field_atomic)

    coeffs = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)

    print("coeffs", coeffs.shape)
    return coeffs


def direct_potential_atomic_to_ao(
    mol, field_atomic, grid_atomic_points, grid_atomic_weights
):
    '''
    Function returns the <mu|field_atomic|nu> matrix where field_atomic is a potential on atomic grid.

    Args:
        mol: PySCF mol class
        field_atomic: direct field-like object given as numpy.array on an atomic grid
        grid_atomic_points: numpy.array of size (npoints,3)
        grid_atomic_weights: numpy.array of size (npoints)

    Returns:
        mat: numpy.array of size (nao,nao)
    '''

    from pyscf.dft.gen_grid import BLKSIZE
    from pyscf.dft.numint import _dot_ao_ao, _dot_ao_dm, _scale_ao, eval_ao

    ao = eval_ao(mol, grid_atomic_points, deriv=0)
    ngrids, nao = ao.shape
    non0tab = np.ones(((ngrids + BLKSIZE - 1) // BLKSIZE, mol.nbas), dtype=np.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    aow = _scale_ao(ao, grid_atomic_weights * field_atomic)
    mat = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)

    # symmetrize the matrix
    mat = mat + mat.T.conj()
    return mat


def direct_to_atomic_accurate(field, othergrid):
    '''From cartesian to atomic grid - accurate algorithm
        This function is optimized for accuracy not for speed.
        Splines field onto othergrid points. Only useable in serial.
    Args:
        field: QMultiPy direct field object
        othergrid: np.array with x,y,z coordinates
    Returns:
        otherfield: np.array with values of the field at the atomic grid points
    '''

    if othergrid.ndim > 1:
        points = othergrid.copy()
    else:
        points = np.array([othergrid]).copy()

    if field.spl_coeffs is None:
        field._calc_spline()

    # Transform othergrid coordinates to the field's grid index space
    metric = np.dot(field.grid.lattice, field.grid.lattice.T)
    ll = np.sqrt(np.diag(metric))

    for i in range(3):
        points[:, i] /= ll[i]  # Divide each coordinate by the lattice parameters
        points[:, i] *= field.grid.nr[
            i
        ]  # + field.spl_order # (we are using padded arrays) Multiply each coordinate by the Grid points for each direction

    mask = np.all((points >= 0) & (points < field.grid.nr + field.spl_order), axis=1)
    otherfield = np.zeros(points.shape[0])
    otherfield[mask] = ndimage.map_coordinates(
        field.spl_coeffs,
        [points[mask, 0], points[mask, 1], points[mask, 2]],
        mode="constant",
    )
    return otherfield


def direct_to_atomic_fast(field, othergrid):
    '''From cartesian to atomic grid - fast algorithm
        This function is optimized for speed not for accuracy.
        It will return the nearest value of the field on the atomic grid.
        Only useable in serial.
    Args:
        field: QMultiPy direct field object
        othergrid: np.array with x,y,z coordinates
    Returns:
        otherfield: np.array with values of the field at the atomic grid points
    '''

    if othergrid.ndim > 1:
        points = othergrid.copy()
    else:
        points = np.array([othergrid]).copy()

    metric = np.dot(field.grid.lattice, field.grid.lattice.T)
    ll = np.sqrt(np.diag(metric))

    for i in range(3):
        points[:, i] /= ll[i]  # Divide each coordinate by the lattice parameters
        points[:, i] *= field.grid.nr[
            i
        ]  # Multiply each coordinate by the Grid points for each direction.
    points = (np.rint(points)).astype(int)  # Round coordinates to the nearest integer
    # Apply modulo for each dimension separately
    p2 = np.zeros(points.shape, dtype=int)
    for i in range(3):
        p2[:, i] = np.mod(points[:, i], field.grid.nr[i])
    mask = np.all((points >= 0) & (points < field.grid.nr), axis=1)
    otherfield = np.zeros(points.shape[0])
    otherfield[mask] = field[
        p2[mask, 0], p2[mask, 1], p2[mask, 2]
    ]  # Getting the nearest point among the Grid and the Coord (The values of the field)
    return otherfield
