"""
Scritpt responsinble to run the simulations for the 8-chain geometry
"""
import utils.pre_processing as pre
import numpy as np
from pathlib import Path

def main():
    # Declare chain and network parameters
    params = (1, 100) ## Kuhn length (nm) and Number of Kuhn segments
    model = '1' ## chain model 
    
    # Declare path to access folder with geometries and do checks
    path_to_geometry_file = Path('..//Geometries//8chain.txt');
    if not path_to_geometry_file.is_file():
        ## Create 8chain geometry
        pre.generate_8chain_geometry(params[1])
    
    # Create filler sphere
    filler_radius = 0.2;
    Nodes, Bonds, Boundary = pre.create_filler_8chain(filler_radius);
    BondTypes = {idx: params[1] for idx in Bonds.keys()} ## chain lengths
    
    # Write lammps data file
    pre.writePositions("filler.dat", Nodes, Bonds, Boundary, BondTypes, model, params)
    breakpoint()
    
    
    
    return


if __name__ == "__main__":
    main()