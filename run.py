"""
Scritpt responsinble to run the simulations for the 8-chain geometry
"""
import utils.pre_processing as pre
import numpy as np
from pathlib import Path

def main():
    # Declare chain and network parameters
    bKuhn, NKuhn = (1, 100) ## Kuhn length (nm) and Number of Kuhn segments
    
    # Declare path to access folder with geometries and do checks
    path_to_geometry_file = Path('..//Geometries//8chain.txt');
    if not path_to_geometry_file.is_file():
        ## Create 8chain geometry
        pre.generate_8chain_geometry(NKuhn)
    
    breakpoint()
    
    
    
    return


if __name__ == "__main__":
    main()