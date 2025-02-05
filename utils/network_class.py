"""
Script defining the NetworkClass.
"""

import numpy as np

class NetworkClass:
    """
    A class to assist querying information from discrete networks.
    """

    def __init__(self, data_file, dump_file):
        """
        Class constructor
        
        Inputs:
            data_file (str): name of LAMMPS data file
            dump_file (str): name of LAMMPS dump file
        """
        self.data_file = data_file
        self.dump_file = dump_file
        
    
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
        Get bonds number of regular chains. 
        
        Inputs:
            None
            
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
