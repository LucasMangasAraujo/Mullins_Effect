"""
Script containing pre-processing functions, such as generation of data files,
reading of input files, etc. See description of each function.
"""
import numpy as np
from pathlib import Path

def create_sphere_points(radius, nPoints):
    """
    Create points on a sphere
    
    Inputs:
        radius: sphere radius
        
    """
    
    
    
    
    
    
    return


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
        f.write("$Bondtypes\n")
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
