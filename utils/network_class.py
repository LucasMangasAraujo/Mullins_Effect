"""
Script defining the NetworkClass.
"""

import numpy as np
import networkx as nx
from collections import defaultdict

class NetworkClass:
    """
    A class to assist querying information from discrete networks. 
    This is the parent class, defining methods that are general for
    all types of networks
    """

    def __init__(self, data_file, dump_file, input_file):
        """
        Class constructor
        
        Inputs:
            data_file (str): name of LAMMPS data file
            dump_file (str): name of LAMMPS dump file
            input_file (str): name of LAMMPS inpute file
            
        """
        self.data_file = data_file
        self.dump_file = dump_file
        self.input_file = input_file
    
    
    def get_computational_params(self, params):
        """
        Get computational params used in the simulation.
        Inputs:
            params (tuple): parameters in with physical units when relevant. 
                            The order is the following:
                                bKuhn: Kuhn length in nm
                                NKuhn: Number of Kuhn segments in the chain.
                                nub3: Normalised chain density in bKuhn3 units.
        
        Outputs:
            computational_params (tuple): parameters in computational units.
        """
        # Extract information DN structure
        Nodes, Bonds = self.get_nodes_and_bonds()
        Boundary = self.get_boundary()
        
        # Unpack input params 
        bKuhn, NKuhn, nub3 = params
        
        # Normalise using the density of crosslinks (subtracting bounary nodes)
        crosslinks = len(Nodes) - len(Boundary)
        upsilonb3 = nub3 / 2
        computational_bKuhn = np.power( upsilonb3 / crosslinks, 1/3)
        
        # Assemple computational_params tuple
        computational_params = (computational_bKuhn, NKuhn)
        
        return computational_params
    
    
    def create_DN_graph(self):
        """
        Turn DN into Graph.
        
        Inputs:
            None
            
        Outputs:
            G (networkx Graph): networkx graph object.
        """
        # Get Network structure
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Create Graph
        G = nx.Graph();
        G.add_nodes_from(Nodes);
        for idx, (n1,n2) in Bonds.items():
            G.add_edge(n1,n2);
        
        return G

        
        return G
        
    def get_boundary(self):
        
        # Read main file to find idx of the boundary nodes
        with open(self.input_file, "r") as f:
            ## Read file until group of boundary nodes is found
            key = f.readline()
            while 'group' not in key:
                key = f.readline()
            
            ## Loop over split line and store ids of boundary nodes
            data = key.strip('\n').split(" ")
            Boundary = []
            for idx in data:
                try:
                    Boundary.append(int(idx))
                except ValueError:
                    continue
                
            
            
        
        return tuple(Boundary)
    
    def get_stretches(self, initial_distances):
        """
        Get chain streches of regular chains
        
        """
        
        # Get current distances
        distances = self.get_distances()
        
        # Perform calculations
        stretches = {idx: distances[idx] / r0 for idx, r0 in initial_distances.items()}
        
        return stretches
    
    def get_distances(self):
        """
        Get end-to-end distance of the chains at a given configuration.
        
        Inputs: 
            None
            
        Outputs:
            distances (dict): distances of regular chains.
        """
        
        # Get Node coordinates and their positions
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Scan and store results
        distances = {}
        for idx, bond in Bonds.items():
            n1, n2 = bond
            vector = Nodes[n1] - Nodes[n2]
            distances[idx] = np.linalg.norm(vector)
        
        
        return distances

    def get_nodes_and_bonds(self):
        """
        Get the bonds and node coordinates of the network.
        Inputs:
            None
            
        Outputs:
            Nodes (dict): dictionary containing node positions
            Bonds (dict): dictionary containing bonds
        """
        # Get data file containing network 
        filename = self.data_file
        
        # Initialize dict
        Bonds = {}
        Nodes = {}
        
        with open(filename, 'r') as f:
            key = f.readline().strip()

            while 'Atoms' not in key:
                key = f.readline().strip()

            f.readline()  # Skip header

            data = f.readline().split()
            while len(data) > 1:
                idx = int(data[0])
                x = float(data[3])
                y = float(data[4])
                z = float(data[5])
                Nodes[idx] = np.array([x, y, z]) 

                data = f.readline().split()
            
            key = f.readline().strip()

            while 'Bonds' not in key:
                key = f.readline().strip()

            f.readline()  # Skip header
            
            data = f.readline().split()
            while len(data) > 1:
                idx = int(data[0])
                n1 = int(data[2])
                n2 = int(data[3])
                Bonds[idx] = [n1, n2] 

                data = f.readline().split()
        
        return Nodes, Bonds

    def calculate_stress(self, dim):
        """
        Calculate stress using virtual work.
        
        Inputs:
            dim (int): problem dimension
        
        Outputs:
            S (ndarray): 3x1 array with the principal stresses
        """
        S = np.zeros(3)
        
        with open(self.dump_file, 'r') as f:
            key = f.readline().strip()

            while 'id' not in key:
                key = f.readline().strip()
            
            data = f.readline().split()
            while len(data) > 1:
                if dim == 3:
                    x = float(data[2])
                    y = float(data[3])
                    z = float(data[4])
                    fx = -float(data[5])  # minus sign for reaction force
                    fy = -float(data[6])
                    fz = -float(data[7])

                    S[0] += fx * x
                    S[1] += fy * y
                    S[2] += fz * z

                else:
                    x = float(data[2])
                    y = float(data[3])
                    fx = -float(data[4])  
                    fy = -float(data[5])

                    S[0] += fx * x
                    S[1] += fy * y
                    
                data = f.readline().split()

        return S
    
    @staticmethod
    def render_stress_units(stress_array, bKuhn, T = 298):
        """
        Dimensionalise dimenionless stress in kPa
        
        Inputs:
            stress_array (ndarray): rubbery components of stress in b^3/kT units
            bKuhn (float): Kuhn length in nm.
            T (float, default = 298 K): temperature in Kelvin.
            
        Ouputs:
            stress_kPa (ndarray): stress array in kPa.
            
        """
        # Declare Boltzmann constant
        kB = 1.380649e-23
        kT = kB * T ## temperatur in energy units
        
        # Render stress array with J/nm3 units
        stress_J_over_nm3 = stress_array * kT / np.power(bKuhn, 3)
        
        # Convert to kPa
        stress_kPa = stress_J_over_nm3 * 1e24
        return stress_kPa


class FillerNetworkClass(NetworkClass):
    """
    A class for filled networks inherented from the NetworkClass
    """
    
    
    #def sphere_overlap(self, placed_spheres):
        
        
        
        
    
    
    
    def get_distances_filler(self):
        """
        Get distances in the network considering the presence 
        of filler partiplces
        
        Inputs:
            None
        
        Outputs:
            distances (dict): distances of regular chains.
        
        """
        
        # Get Node coordinates and their positions
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Get which bonds are regular
        idx_of_regular_bonds = self.get_regular_bonds()
        filtered_Bonds = {idx: bond for idx, bond in Bonds.items() if idx in idx_of_regular_bonds}
        
        # Scan and store results
        distances = {}
        for idx, bond in filtered_Bonds.items():
            n1, n2 = bond
            vector = Nodes[n1] - Nodes[n2]
            distances[idx] = np.linalg.norm(vector)
        
        return distances
    
    
    def get_regular_bonds(self):
        """
        Get bonds number of regular chains, i.e, not forming filler particles.
        
        Inputs:
            None
            
        Outputs:
            idx_of_regular_bonds (tuple): sequence of indices of regular chains.
        """
        
        # Read file and find the regular bonds
        idx_of_regular_bonds = []
        with open(self.data_file, "r") as f:
            ## Read until spring coefficients are found
            key = f.readline()
            while "Bond Coeffs" not in key:
                key = f.readline()
            
            f.readline() ## empty line
            
            ## Check if hybrid bond style is being used
            data = f.readline().strip("\n").split(" ")
            try:
                float(data[1])
                hybrid_style_flag = False
            except ValueError:
                hybrid_style_flag = True
            
            ## Find the bonds defining the sphere
            while len(data) > 1:
                
                if hybrid_style_flag:
                    if not 'harmonic' in data:
                        idx_of_regular_bonds.append(int(data[0]))
                else:
                    if np.isclose(float(data[-1]), 0):
                        idx_of_regular_bonds.append(int(data[0]))
                    
                data = f.readline().strip("\n").split(" ")
            
        
        return idx_of_regular_bonds
    
    
    
    
    def get_volume_fraction(nFillers, filler_radius):
        """
        Calculate the volume fraction in the network for a given number of 
        particles and dimensionless filler radius.
        
        Inputs:
            nFillers(int): number of filler networks in the network
            filler_radius (float):
        Outputs:
            
        
        """
        # Calculate the volume fraction
        vol_fraction = 4 * np.pi * np.power(filler_radius, 3) / 3.
        return vol_fraction
    
    
    
    def get_filler_angles_deviations(self, angle_to_pair):
        """
        Get deviation of the filler angles after equilibrium for a 
        given deformation.
        
        Inputs:
            angle_to_pair(dict): angle-to-bond pair map.
            
        Outputs:
            deviations (dict): average and std current-to-initial angle ratio
        
        """
        # Query DN
        Nodes, Bonds = self.get_nodes_and_bonds()
        Angles = self.get_angles_and_triplets()
        
        # Loop over the angle-to-pair map
        temp = defaultdict(list)
        for angle_idx, bonds_idx in angle_to_pair.items():
            ## Get the vectors
            b1, b2 = Bonds[bonds_idx[0]], Bonds[bonds_idx[1]]
            v1 = Nodes[b1[0]] - Nodes[b1[1]]
            v2 = Nodes[b2[0]] - Nodes[b1[1]]
            
            ## Calculate current angle between pairs
            dot_product = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
            theta = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
            
            ## Store in appropriate position the information
            b1, b2 = set(b1), set(b2)
            node = next(iter((b1.intersection(b2))))
            theta0 = Angles[angle_idx][1]
            temp[node].append(np.abs(theta - theta0))
            
        
        # Use the average the results in the dict
        deviations = {key: (np.mean(data), np.std(data)) for key, data in temp.items()}
        
        # Return also maxmum and minimum deviation
        max_key = max(deviations, key=lambda k: deviations[k][0])
        min_key = min(deviations, key=lambda k: deviations[k][0])
        max_avg_deviation = deviations[max_key][0]
        min_avg_deviation = deviations[min_key][0]
        
        return deviations, max_avg_deviation, min_avg_deviation
    
    def get_filler_radii_deviations(self, bond_flags, selected_nodes, filler_radius):
        """
        Get the deviation in filler radii after equilibrium for a given
        deformation.
        
        Inputs:
            bond_flags (dict): flags that inform if bond forms a filler.
            selected_nodes (list): indiced of the nodes acting as the filler centres.
            filler_radius (flaot): expected radius of the filler.
            
        Outputs:
            deviations (list): list of current-to-initial radii ratios.
        """
        # Query DN
        Nodes, Bonds = self.get_nodes_and_bonds()
        
        # Map filler bonds to their centres
        bond_to_centre_map = defaultdict(list)
        for idx, bond in Bonds.items():
            if all(bond_flags[idx]):
                for node in bond:
                    if node in selected_nodes:
                        bond_to_centre_map[node].append(bond)
            
        
        
        # Calculate deviations
        deviations = [
            np.mean([np.linalg.norm(Nodes[n1] - Nodes[n2]) for (n1, n2) in bond_to_centre_map[node]]) / filler_radius
            for node in selected_nodes
        ]
    
        return deviations
    
    
    
    def get_angles_and_triplets(self):
        """
        Get initial angles and the triplets defining each one of them
        Inputs:
            None
        
        Outputs:
            Angles (dict): triplet of nodes forming the angle and their 
                           initial values. The function will return None
                           if there are not angle restrictions.
        """
        # Check if method is aplicable 
        with open(self.data_file, "r") as f:
            lines = f.readlines()
        NA_flags = ["Angles" in line or "Angle Coeffs" in line for line in lines]
        if not any(NA_flags):
            return None
        
        # Scan file
        with open(self.data_file, "r") as f:
            ## Go through file until angle coefficientes were found
            key = f.readline()
            while "Angle Coeffs" not in key:
                key = f.readline()
            
            f.readline() ##read empty line
            
            ## Read the theta0 values of each angle
            theta0 = {}
            data = f.readline().strip("\n").split(" ")
            while len(data) > 1:
                theta0[int(data[0])] = float(data[2])
                data = f.readline().strip("\n").split(" ")
            
            ## Now scan until the angles section is reached
            key = f.readline()
            while "Angle" not in key:
                key = f.readline()
            
            f.readline() ## empty line 
            
            ## Read list of triplets and to which angle they are linked
            data = f.readline().strip("\n").split(" ")
            triplets = {}
            while len(data) > 1:
                idx = int(data[1])
                triplet = int(data[2]), int(data[3]), int(data[4])
                triplets[idx] = triplet
                data = f.readline().strip("\n").split(" ")
            
        
        # Assamble list of angles
        Angles = {}
        for idx in theta0.keys():
            Angles[idx] = triplets[idx], theta0[idx]
        
        return Angles
        
    
    @staticmethod
    def estimate_nFillers(vol_fraction, filler_radius):
        """
        Estimate the number of fillers particles needed in the network
        for a given volume fraction of fillers and normalize filler 
        radius.
        """
        nFillers = np.ceil(3 * vol_fraction / (4 * np.pi * np.power(filler_radius, 3)))
        return int(nFillers)
    
    @staticmethod
    def estimated_filler_radius(vol_fraction, nFillers):
        """
        Estimate normalised filler radius for a given filler volume
        fraction and number of filler particles in the network
        """
        filler_radius = np.power(3 * vol_fraction / (4 * np.pi * nFillers), 1 / 3)
        return filler_radius