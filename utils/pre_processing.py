"""
Script containing pre-processing functions, such as generation of data files,
reading of input files, etc. See description of each function.
"""
import numpy as np
from pathlib import Path


def create_filler_8chain(filler_radius):
    """
    Create filler particle in the 8-chain unit cell, and add them to the Nodes 
    and Bonds dict
    
    Inputs:
        filler_radius (float): Radius of the filler particle
        
    Returns:
        new_Nodes (dict): Dict containing the old coordinates plus new ones from the particle
        new_Bonds (dict): Dict with new connections in the unit cell
        Boundary (list): List with strings containing the ids of nodes at the boundary
    """
    
    # Read file with 8-chain geometry
    out = readGeometry('..//Geometries//8chain.txt')
    Nodes, Bonds = out[0], out[1]
    Boundary = out[2]
    nNodes, nBonds = len(Nodes), len(Bonds)
    
    # Create filler as a sphere with given radius in the centre of the unit cell
    filler_points, filler_bonds = create_sphere_points_8chain(Nodes = Nodes, radius = filler_radius);
    nFillerPoints, nFillerBonds = len(filler_points), len(filler_bonds)
    
    # Creat pertued filler points
    perturbed_filler_points, perturbed_filler_bonds = perturb_sphere_points_8chain(filler_points, Nodes[9], nNodes)
    nPerturbedPoints, nPerturbedBonds = len(perturbed_filler_points), len(perturbed_filler_bonds);
    
    # Add new filler_points to the dict of nodes
    old_node_keys = set(Nodes.keys()) ## store old keys of nodes dict
    new_Nodes = {}
    nNewNodes = nNodes + nFillerPoints + nPerturbedPoints;
    for idx in range(nNewNodes):
        if idx + 1 in old_node_keys:
            new_Nodes[idx + 1] = Nodes[idx + 1]
        elif idx + 1 <= nNodes + nFillerPoints:
            new_Nodes[idx + 1] = filler_points[idx - nNodes, :]
        else:
            new_Nodes[idx + 1] = perturbed_filler_points[idx - nNodes - nFillerPoints, :]
    
    
    # Form new dict of bonds
    aux = list(filler_bonds) + list(perturbed_filler_bonds)
    new_Bonds = {idx + 1: bond for idx, bond in enumerate(aux)}
    nNewBonds = len(new_Bonds)
    
    # Assign bond types to each newly craeted bond
    bond_flags = {};
    old_node_keys.remove(9) ## remove numbering of the central node
    for idx, bond in new_Bonds.items():
        if any(n in old_node_keys for n in bond):
            ## Old connections
            bond_flags[idx] = False
        else:
            ## Connections between pertubations and the remaining points
            bond_flags[idx] = True
        
    
    return new_Nodes, new_Bonds, Boundary, bond_flags

def create_sphere_points_8chain(Nodes, radius):
    """
    Create points on a sphere for the 8chain geometry, and new connections to be placed
    
    Inputs:
        Nodes (dict): Dict containing the coordinates of the nodes in the 8-chain cell
        radius (float): sphere radius.
        
    Outputs:
        sphere_points (ndarray): coordinates of the points on the sphere
        sphere_bonds (list): connections between the points on the sphere and other points 
                             in the cell.
    """
    
    # Create Nx3 matrix with unit cell vertices coords
    cube_vertices = np.array(tuple(Nodes.values())[:-1]);
    sphere_centre = Nodes[9] ## cell centre
    nVertices = len(cube_vertices)
    
    # Create vectors
    vectors_to_vertices = cube_vertices - sphere_centre ## from sphere centre to cube corners
    unit_vectors_to_vertices = vectors_to_vertices / \
                                np.linalg.norm(vectors_to_vertices, axis = 1)[:, np.newaxis]
    
    # Scale unit vectors by the sphere radius
    sphere_points = (unit_vectors_to_vertices * radius) + sphere_centre
    
    # Create numbering (1-indexed) of vertices and points on the sphere
    vertices_numbering = tuple(Nodes.keys())
    sphere_points_numbering = np.arange(nVertices + 2, nVertices + len(sphere_points) + 2, 1)
    
    # Create bonds between the sphere points and its centre
    sphere_bonds = [] ## list of bonds (tuples) initialization
    for i in range(len(sphere_points)):
        bond = sphere_points_numbering[i], 9
        sphere_bonds.append(bond)
    
    # Loop the sphere points to find correspondances between sphere points and vertices
    for i, sphere_point in enumerate(sphere_points):
        ## Get unit vector from centre to point on the sphere
        vector_to_sphere = sphere_point - sphere_centre
        unit_vector_to_sphere = vector_to_sphere / np.linalg.norm(vector_to_sphere)
        
        ## Find vertext that best aligns with the uni vector
        dot_products = np.dot(unit_vectors_to_vertices, unit_vector_to_sphere) ## matrix with dot products
        matching_vertex_idx = np.argmax(dot_products)
        
        ## Create bond between point and corresponding vertices
        bond = sphere_points_numbering[matching_vertex_idx], vertices_numbering[matching_vertex_idx]
        #sphere_bonds.append(bond)
    
    
    return sphere_points, sphere_bonds

def perturb_sphere_points_8chain(sphere_points, sphere_centre, nNodes, epsilon = 1e-4):
    """
    Create points that pertubed versions of the points on the spheres
    
    Inputs:
        sphere_points (ndarray): Nx3 array with the coordinates of points on the sphere.
        sphere_centre (ndarray:): 3-row array with coordinates of the sphere centre.
        nNodes (int): Original number of nodes in the network.
        epsilon (float, optional): Magnitude of the pertubation.
        
    Outputs:
        perturbed_sphere_points (nparray): Nx3 array with coordinates of perturbed points.
        
    
    """
    # Calculate unit vectors pointing in the radial direction
    vector = sphere_points - sphere_centre
    unit_vectors = vector / np.linalg.norm(vector, axis = 1)[:, np.newaxis]
    
    # Created perturbed points
    perturbed_sphere_points = sphere_points + (epsilon * unit_vectors)
    nPerturbed = len(perturbed_sphere_points);
    
    # Create arrays with global node numbering
    vertices_global_numbering = np.arange(1, nNodes) ## discard sphere centre
    sphere_global_numbering = np.arange(nNodes + 1, nNodes + len(sphere_points) + 1, 1)
    
    # Create bonds sphere-perturbed 
    perturbed_bonds = [];
    for i, node_idx in enumerate(sphere_global_numbering):
        ## Create first sphere-to-pertubation bond
        bond = node_idx + nPerturbed, node_idx
        perturbed_bonds.append(bond)
        
        ## Create now pertubation-to-vertex bond
        bond = node_idx + nPerturbed, vertices_global_numbering[i]
        perturbed_bonds.append(bond)
        
    
    return perturbed_sphere_points, perturbed_bonds

def generate_8chain_geometry(NKuhn, dim = 3):
    """
    Generate geometry file containing unit cell of the 8-chain model
    Inputs:
        NKuhn (float): Number of Kuhn segments in the chain needed for the reading processing
        dim (integer): Optional integer indicating the dimension of the problem (optional)
        
    """
    # Create path to folder that will contain the geomtry file
    parent_dir = Path.cwd().parent
    geom_folder = parent_dir / "Geometries"
    geom_folder.mkdir(exist_ok = True); ## makes sure that the folder exists
    
    # Create node positions
    Nodes = {};
    Nodes[1] = np.array([0.0, 0.0, 0.0]);
    Nodes[2] = np.array([1.0, 0.0, 0.0]);
    Nodes[3] = np.array([1.0, 1.0, 0.0]);
    Nodes[4] = np.array([0.0, 1.0, 0.0]);
    Nodes[5] = np.array([0.0, 0.0, 1.0]);
    Nodes[6] = np.array([1.0, 0.0, 1.0]);
    Nodes[7] = np.array([1.0, 1.0, 1.0]);
    Nodes[8] = np.array([0.0, 1.0, 1.0]);
    Nodes[9] = np.array([0.5, 0.5, 0.5]);
    
    # Create connections dict
    Bonds = {};
    Bonds[1] = (9, 1);
    Bonds[2] = (9, 2);
    Bonds[3] = (9, 3);
    Bonds[4] = (9, 4);
    Bonds[5] = (9, 5);
    Bonds[6] = (9, 6);
    Bonds[7] = (9, 7);
    Bonds[8] = (9, 8);
    
    # Create list of boundary nodes
    Boundary = np.arange(1, 9, 1, dtype = np.int16)
    
    # Create bond types
    Bond_types = {idx: NKuhn for idx in Bonds.keys()}
    
    # Write geometry file
    with open(geom_folder / "8chain.txt", "w+") as f:
        ## Start with nodes coordinates
        f.write("$nodes\n")
        for idx, coord in Nodes.items():
            aux = (idx, coord[0], coord[1],  coord[2]) ## auxiliar tuple for writting
            f.write("%d, %g, %g, %g\n" %aux)
        ## Write connections now
        f.write("$bonds\n")
        for idx, (n1, n2) in Bonds.items():
            aux = idx, n1, n2
            f.write("%d, %d, %d\n" %aux);
        ## Write Boundary
        f.write("$boundary\n")
        for node in Boundary:
            f.write('%d ' %node);
        f.write("\n");
        ## Write chain length distribution
        f.write("$BondTypes\n")
        for idx, chain_length in Bond_types.items():
            aux = idx, chain_length
            f.write("%d, %g\n" %aux);
    
    
    return

def writePositions(filename, Nodes, Bonds, Boundary, BondTypes, model, params):

    """
    Writes a file containing the architecture of the discrete network 
    and the parameters of each chain in it in a way that LAMMPS can 
    read it and proceed with the energy minimisation process.
    
    
    filename : name of the file that will be generated
    Nodes : dictionary whose keys are the IDs of the nodes,
            and the values are a list with node coordinates
    Bonds: dictionary whose keys are the bond IDs and the,
            values are a list containing the pair of nodes 
            connected.
    Boundary: list of strings with the IDs of the boundary nodes
    BondTypes: dictionary whose keys are the bond IDs and the,
            values are the chain lengths.
    model : string indicating the type of bond behaviour.
            model = '1': Gaussian
            model = '2': FJC
            model = '3': Breakable extensible FJC
            model = '4': Breakable FJC
            model = '5': Harmonic (Hookean)
            molde = '6': Breakable Gaussian chain
    params: list containing chain parameters other than 
            the chain length.
            
    
    The function returns None.
    
    """
    
    # Calculate the number of nodes, bonds, and boundary nodes
    Natoms = len(Nodes);
    Nbonds = len(Bonds);
    Nboundary = len(Boundary);
    NbondTypes = len(BondTypes);
    
    # Check for polydispersity
    chain_lengths = np.array(list(BondTypes.values()));
    polydispersity_flag = not np.all(chain_lengths == chain_lengths[0])
    if not polydispersity_flag: NbondTypes = 1;
    
    # Open file and write on it
    with open(filename, 'w') as f:
        
        #Header
        f.write('LAMMPS data file for the initial network geometry\n\n');

        #Number of nodes and bonds
        f.write('%d atoms\n' %Natoms);
        f.write('1 atom types\n');
        f.write('%d bonds\n' %Nbonds);
        f.write('%d bond types\n\n' %NbondTypes);

        #Box dimensions
        f.write('-0.1 1.1 xlo xhi\n');
        f.write('-0.1 1.1 ylo yhi\n');
        f.write('-0.1 1.1 zlo zhi\n\n');

        #Masses
        f.write('Masses\n\n1 1\n\n');

        #Bond coefficients
        f.write('Bond Coeffs\n\n');
        bKuhn = params[0]; ## Kuhn length
        
        if polydispersity_flag:
            for idx, N in BondTypes.items():
                
                if model in ['1', '5']: ## Gaussian chain (harmonic)
                    kappa = (3./2.) * (1 / ( N * pow(bKuhn, 2) )); ## Bond stiffness in the Gaussian regime
                    if model =='1':
                        f.write('%d %g %g\n'%(idx, kappa, 0.)); ## zero rest length
                    elif model == '6':
                        f.write('%d %g %g %g\n'%(idx, kappa, 0., N * bKuhn)); ## Add the contour length
                    else:
                        rest_length = params[2];
                        f.write('%d %g %g\n'%(idx, kappa, rest_length)); ## zero rest length
                    
                elif model == '2' or model == '4': ## FJC or breakable FJC
                    f.write('%d %g %g\n' %(idx, bKuhn, N));
                
                elif model == '3': ## Extensible FJC
                    bKuhn, Eb, critical_eng = tuple(params);
                    f.write('%d %g %g %g %g\n' %(idx, bKuhn, N, Eb, critical_eng));
                
            
        else:
            N = chain_lengths[0];
            
            if model in ['1', '5', '6']: ## Gaussian chain (harmonic)
                
                kappa = (3./2.) * (1 / ( N * pow(bKuhn, 2) )); ## Bond stiffness in the Gaussian regime
                if model == '1':
                    f.write('1 %g %g\n'%(kappa, 0.)); ## zero rest length
                elif model == '6':
                    f.write('1 %g %g %g\n'%(kappa, 0., N * bKuhn)); ## Add the contour length
                else:
                    rest_length = params[2];
                    f.write('1 %g %g\n'%(kappa, rest_length)); ## zero rest length
                
            elif model == '2' or model == '4': ## FJC or breakable FJC
                f.write('1 %g %g\n' %(bKuhn, N));
            
            elif model == '3': ## Extensible FJC
                bKuhn, Eb, critical_eng = tuple(params);
                f.write('1 %g %g %g %g\n' %(bKuhn, N, Eb, critical_eng));
            
            
        f.write('\n\n');
        
        #Atoms ids and positions
        f.write('Atoms\n\n')
        for idx in Nodes:
            f.write('%d 1 1 %g %g %g\n' %(idx,Nodes[idx][0],Nodes[idx][1],Nodes[idx][2]))
        
        f.write('\n')
        # Bonds IDs and pairs of nodes connected by each bond
        f.write('Bonds\n\n') 

        # Check for polydispersity
        for idx in Bonds:
            if(polydispersity_flag): ## Each bond has its own type
                f.write('%d %d %g %g\n' % (idx,idx,Bonds[idx][0],Bonds[idx][1]) );
            else: ## there is one bond type only
                f.write('%d 1 %g %g\n' %(idx,Bonds[idx][0],Bonds[idx][1]));

        f.write('\n')

    return


def readBondTypes(input,Nbonds):
    '''
        This fucntion reads the bond types and store them in dictionary 
        called BondTypes when there are more than one bond type
        -------------------------------------------------------------
        Inputs:
            input: File pointer to be read;
            Nbonds: Number of Bonds in the network;
        *************************************************************
            Outputs
            BondTypes: Dic with bonds types
            key: previous line read
    '''
    #================================================================
    ## Variable Declaration
    # Constant Parameters
    R0 = float(0.); R1 = float(1.0);
    # Logicals 
    again = False;
    # Dictionaryes
    BondTypes = {};
    #================================================================
    # Start scannig Process Depending 
    for i in range(1,Nbonds + 1):
        key = input.readline().strip('\n');
        data = key.split();
        BondTypes[i] = float(data[1]);
    
    return BondTypes, key


def readBoundary(input):

    """
    Read the list of boundary nodes from a file with pointer input
    and store them in an array
    """
    key = input.readline().strip(' \n') #read the next line in the input file
    if ("," in key):
        data = key.split(',')
    else:
        data = key.split(' ');

    return data,key

def readBonds(input):

    """
    Read the list of bonds from a file with pointer input
    and store them in a dictionary
    """

    again = True
    bond_dict = {}

    while again is True:
        key = input.readline().strip('\n')  #read the next line in the input file
        if '$' in key:
            again = False
        else:
            data = key.split(',')
            bond_dict[int(data[0])] = [int(data[1]),int(data[2])]

    return bond_dict,key

def readNodes(input):

    """
    Read the list of node from a file with pointer input
    and store them in a dictionary
    """

    #read nodes
    again = True
    node_dict = {}

    while again is True:

        #read the next line in the input file
        key = input.readline().strip('\n')

        if '$' in key:
            again = False
        else:
            data = key.split(',')
            x = float(data[1])
            y = float(data[2])
            z = float(data[3])
            node_dict[int(data[0])] = [float(x),float(y),float(z)]

    return node_dict,key

def readGeometry(filename):

    """ 
    Read network geometry file
    """

    f = open(filename,'r')

    again = True
    key = f.readline().strip('\n')  #read the next line in the input file

    while again is True:

        if 'nodes' in key:
            Nodes,key = readNodes(f)

        elif 'bonds' in key:
            Bonds, key = readBonds(f)

        elif 'boundary' in key:
            Boundary, key = readBoundary(f)
            key = f.readline().strip('\n') # Start Reading Again $
        elif 'BondTypes' in key:
            BondTypes, key = readBondTypes(f,len(Bonds));
        else:
            again = False

    f.close()

    return Nodes,Bonds,Boundary,BondTypes




if __name__ == "__main__":
    main()
