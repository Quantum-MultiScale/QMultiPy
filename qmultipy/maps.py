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
    v_H = fg.ifft(force_real=True)
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
    # Use pseudoinverse since eri matrix may be singular
    #coeffs = np.linalg.pinv(eri.transpose(0,2,1,3).reshape(N,N)) @ mat.flatten()
    coeffs = np.linalg.pinv(eri.reshape(N,N)) @ mat.flatten()

    # check if field is well represented
    ovlp = mol.intor("int1e_ovlp").flatten()
    int_field = field.integral()
    int_mat = np.einsum("m,m->", coeffs, ovlp)

    if int_mat - int_field > 1e-6:
        print("int_mat",int_mat)
        print("int_field",int_field)
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

    from pyscf.dft.numint import eval_ao, _scale_ao, _dot_ao_ao
    from pyscf.dft.gen_grid import BLKSIZE
    
    ao=eval_ao(mol,grid_atomic_points,deriv=0)
    ngrids, nao = ao.shape
    non0tab = np.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),dtype=np.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    aow = _scale_ao(ao,grid_atomic_weights*field_atomic)

    coeffs = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)

    print("coeffs",coeffs.shape)
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

    aow = _scale_ao(ao, grid_atomic_weights*field_atomic)
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
        points[:, i] *= field.grid.nr[i] #+ field.spl_order # (we are using padded arrays) Multiply each coordinate by the Grid points for each direction

    mask = np.all((points >= 0) & (points < field.grid.nr + field.spl_order), axis=1)
    otherfield = np.zeros(points.shape[0])
    otherfield[mask] = ndimage.map_coordinates(field.spl_coeffs, [points[mask, 0], points[mask, 1], points[mask, 2]], mode="constant")
    return otherfield


def direct_to_atomic_accurate_alternative(field, othergrid):
    ''' From cartesian to atomic grid using Kernel Ridge Regression
        This function uses local KRR for interpolation, which can be more accurate
        than splines for irregular grids.
        
    Args:
        field: QMultiPy direct field object
        othergrid: np.array with x,y,z coordinates
    Returns:
        otherfield: np.array with values of the field at the atomic grid points
    '''
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.neighbors import KDTree
    
    # Convert othergrid to points array
    if othergrid.ndim > 1:
        query_points = othergrid.copy()
    else:
        query_points = np.array([othergrid]).copy()
        
    # Transform query points to the field's grid index space
    metric = np.dot(field.grid.lattice, field.grid.lattice.T)
    ll = np.sqrt(np.diag(metric))
    
    for i in range(3):
        query_points[:, i] /= ll[i]  # Scale by lattice parameters
        query_points[:, i] *= field.grid.nr[i]  # Scale to grid points
        
    # Use DirectGrid to get source points
    grid = field.grid
    
    # Reshape grid.s to get source points
    # grid.s has shape (3, nx, ny, nz), we need to reshape to (nx*ny*nz, 3)
    source_points = np.stack(grid.s, axis=-1)  # Stack along new last axis
    source_points = source_points.reshape(-1, 3)  # Reshape to 2D array
    
    # Convert to index space by multiplying each coordinate by corresponding nr
    for i in range(3):
        source_points[:, i] *= grid.nr[i]
        
    source_values = field.ravel()
    
    # Initialize KDTree for efficient neighbor search
    tree = KDTree(source_points)
    
    # Parameters for local KRR
    n_neighbors = field.spl_order**3
    gamma = 1.0  # RBF kernel parameter
    alpha = 1e-10  # Regularization strength
    
    # Find nearest neighbors for each query point
    distances, indices = tree.query(query_points, k=n_neighbors)
    
    # Initialize output array
    otherfield = np.zeros(query_points.shape[0])
    
    # Perform local KRR for each query point
    for i, query in enumerate(query_points):
        # Get local neighborhood
        local_points = source_points[indices[i]]
        local_values = source_values[indices[i]]
        
        # Skip if all local values are zero
        if np.all(local_values == 0):
            continue
            
        # Fit local KRR model
        krr = KernelRidge(
            kernel='rbf',
            #gamma=gamma,
            alpha=alpha
        )
        krr.fit(local_points, local_values)
        
        # Predict at query point
        pred = krr.predict(query.reshape(1, -1))
        otherfield[i] = pred[0]
        
    # Apply mask for out-of-bounds points
    mask = np.all((query_points >= 0) & (query_points < grid.nr), axis=1)
    otherfield[~mask] = 0
    
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

    if othergrid.ndim > 1:
        points = othergrid.copy()
    else:
        points = np.array([othergrid]).copy()


    metric = np.dot(field.grid.lattice, field.grid.lattice.T)
    ll = np.sqrt(np.diag(metric))

    for i in range(3):
        points[:,i] /= ll[i] #Divide each coordinate by the lattice parameters
        points[:,i] *= field.grid.nr[i] #Multiply each coordinate by the Grid points for each direction.
    points=(np.rint(points)).astype(int) #Round coordinates to the nearest integer
    # Apply modulo for each dimension separately
    p2 = np.zeros(points.shape,dtype=int)
    for i in range(3):
        p2[:,i] = np.mod(points[:,i], field.grid.nr[i])
    mask = np.all((points >= 0) & (points < field.grid.nr), axis=1)
    otherfield = np.zeros(points.shape[0])
    otherfield[mask] = field[p2[mask,0],p2[mask,1],p2[mask,2]] #Getting the nearest point among the Grid and the Coord (The values of the field)
    return otherfield




