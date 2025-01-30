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
                Nodes[idx] = [x, y, z] 

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
