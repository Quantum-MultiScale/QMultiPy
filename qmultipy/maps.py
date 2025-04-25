import warnings
from functools import partial
import numpy as np
from scipy import ndimage
from scipy import signal
from qmultipy.grid import DirectGrid, ReciprocalGrid


def direct_to_otherdirect(field,othergrid):
    # From Yongshuo
    return otherfield



def direct_to_GTOs_full(field,mol,grid_level=4):
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
    v_H = fg.ifft()
    # Get the atomic grid
    grids = Grids(mol)
    grids.level = grid_level
    grids.build()
    # put the field on the atomic grid
    field_atomic = direct_to_atomic_accurate(v_H,grids.coords)
    # get mat_mu nu = <mu nu | 1/r12 | field >
    mat = direct_potential_atomic_to_ao(mol,field_atomic,grids.coords,grids.weights)
    # set up linear system for DF problem: mat = eri * c
    eri = mol.intor("int2e", aosym = "s1")
    n = mat.shape[0]
    N = n**2
    coeffs = np.linalg.solve(eri.reshape(N,N), mat.flatten())    

    # check if field is well represented
    ovlp = mol.intor("int1e_ovlp").flatten()
    int_field = field.integral()
    int_mat = np.einsum("m,m->", coeffs, ovlp)

    if int_mat - int_field > 1e-6:
        warnings.warn("Field is not well represented by GTOs")
    return coeffs.reshape(n,n)



def direct_to_GTOs(field,mol,grid_level=4):
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
    field_atomic = direct_to_atomic_accurate(field,grids.coords)

    # put direct_atomic to AOs
    c = direct_atomic_to_ao(mol,field_atomic,grids.coords,grids.weights)

    # check if field is well represented
    # <mu| field> = sum_nu c_nu * <mu|nu>

    ovlp = mol.intor("int1e_ovlp")
    c_field = np.einsum("mn,mn->", c, ovlp)
    if np.any(np.abs(c - c_field) > 1e-6):
        warnings.warn("Field is not well represented by GTOs")
    return c


def direct_atomic_to_ao(mol,field_atomic,grid_atomic_points,grid_atomic_weights):
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

    from pyscf.dft.numint import eval_ao, _scale_ao, _dot_ao_dm
    from pyscf.dft.gen_grid import BLKSIZE
    
    ao=eval_ao(mol,grid_atomic_points,deriv=0)
    ngrids, nao = ao.shape
    non0tab = np.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),dtype=np.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    aow=np.empty((np.shape(grid_atomic_points[:,0])[0],nao)) 
    mat=np.empty((nao,nao))
    
    
    aow = _scale_ao(ao, grid_atomic_weights*field_atomic)
    coeffs = _dot_ao_dm(mol, ao, aow, non0tab, shls_slice, ao_loc)

    return coeffs


def direct_potential_atomic_to_ao(mol,field_atomic,grid_atomic_points,grid_atomic_weights):
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

    from pyscf.dft.numint import eval_ao, _scale_ao, _dot_ao_ao, _dot_ao_dm
    from pyscf.dft.gen_grid import BLKSIZE
    
    ao=eval_ao(mol,grid_atomic_points,deriv=0)
    ngrids, nao = ao.shape
    non0tab = np.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),dtype=np.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    aow=np.empty((np.shape(grid_atomic_points[:,0])[0],nao)) 
    mat=np.empty((nao,nao))
    
    aow = _scale_ao(ao, grid_atomic_weights*field_atomic, out=aow)
    mat = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)

    # symmetrize the matrix
    mat = mat + mat.T.conj()
    return mat


def direct_to_atomic_accurate(field, othergrid):
    ''' From cartesian to atomic grid - accurate algorithm
        This function is optimized for accuracy not for speed. 
        Splines field onto othergrid points. Only useable in serial.
    Args:
        field: QMultiPy direct field object
        othergrid: np.array with x,y,z coordinates
    Returns:
        otherfield: np.array with values of the field at the atomic grid points
    '''

    if othergrid.ndim > 1 and othergrid.shape[0] > 3:
        points = othergrid.transpose()    
    else:
        points = othergrid

     if field.spl_coeffs is None:
        field._calc_spline()

    otherfield = ndimage.map_coordinates(field.spl_coeffs, [points[:, 0], points[:, 1], points[:, 2]], mode="wrap")
    return otherfield


def direct_to_atomic_fast(field, othergrid):
    ''' From cartesian to atomic grid - fast algorithm
        This function is optimized for speed not for accuracy. 
        It will return the nearest value of the field on the atomic grid.
        Only useable in serial.
    Args:
        field: QMultiPy direct field object
        othergrid: np.array with x,y,z coordinates
    Returns:
        otherfield: np.array with values of the field at the atomic grid points
    '''

    if othergrid.ndim > 1 and othergrid.shape[0] > 3:
        points = othergrid.transpose()    
    else:
        points = othergrid

    metric = np.dot(field.grid.lattice, field.grid.lattice.T)
    ll = np.sqrt(np.diag(metric))

    for i in range(3):
        points[:,i] /= ll[i] #Divide each coordinate by the lattice parameters
        points[:,i] *= field.grid.nr[i] #Multiply each coordinate by the Grid points for each direction.
    p2=(np.rint(points)).astype(int) #Round coordinates to the nearest integer
    p2=np.mod(p2, field.grid.nr) #arr1 % arr2 #Remainder of Div.
    otherfield=field[p2[:,0],p2[:,1],p2[:,2]] #Getting the nearest point among the Grid and the Coord (The values of the field)
    return otherfield




