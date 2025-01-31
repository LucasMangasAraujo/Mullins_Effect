"""
Scritpt responsinble to run the simulations for the 8-chain geometry
"""
import utils.pre_processing as pre
import utils.sim_executor as sim
from utils.network_class import NetworkClass
from utils.loading import create_monotonic_load, deformation_gradient
import numpy as np
from pathlib import Path
import math

def main():
    
    # Declare chain and network parameters
    params = (1, 100) ## Kuhn length (nm) and Number of Kuhn segments
    nub3 = 1e-3; ## normalised (via Kuhn length) chain density
    model = '1' ## chain model 
    dim = 3 ## problem dimension
    computational_bKuhn = pow(params[0] / 8, 1/3) ## normalised Kuhn length
    computational_params = (computational_bKuhn, 100)
    angle_model = '1'
    angle_stiffness = (10, )
    
    # Define load
    loading = 1
    max_stretch = 5
    increments = 4
    stretch_array, stretch_increment = create_monotonic_load(max_stretch, increments)
    
    # Declare path to access folder with geometries and do checks
    path_to_geometry_file = Path('..//Geometries//8chain.txt');
    if not path_to_geometry_file.is_file():
        ## Create 8chain geometry
        pre.generate_8chain_geometry(params[1])
    
    # Create filler sphere
    filler_radius = 0.1;
    Nodes, Bonds, Boundary, bond_flags, Angles = pre.create_filler_8chain(filler_radius);
    BondTypes = {
                idx: params[1] / 100 if bond_flags[idx] else params[1]
                for idx in Bonds.keys()
                } ## chain lengths
    
    # Write lammps data file
    pre.writePositions("filler.dat", Nodes, Bonds, Boundary, BondTypes, model, params)
    pre.write_data_file("filler_angles.dat", Nodes, Bonds, Angles, Boundary, BondTypes, model, 
                        params, angle_model, angle_stiffness)
    
    # run relaxation without angle constraint
    sim.run_relaxation(dim, "filler.dat", Boundary, model)
    DN = NetworkClass("filler.dat", "test.res")
    current_position, current_bonds = DN.get_nodes_and_bonds()
    cauchy_stress = DN.calculate_stress(dim);
    
    # Repeat but know using angle constraints
    sim.run_relaxation_angles(dim, "filler_angles.dat", Boundary, model, angle_model)
    DN_angles = NetworkClass("filler_angles.dat", "test.res")
    current_position_angles, current_bonds_angles = DN_angles.get_nodes_and_bonds()
    cauchy_stress_angles = DN_angles.calculate_stress(dim);
    
    # Apply load
    for i in range(1, increments):
        ## Run deformation step
        F = deformation_gradient(loading, stretch_array[i])
        print("F_11 = %g, F_22 = %g, F_33 = %g" %tuple(F))
        sim.runinc(loading, i + 1, stretch_increment, dim, main_file = 'main.in')
        DN = NetworkClass("filler.dat", "test.res")
        cauchy_stress = DN.calculate_stress(dim);
        print("No angles: S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress))
        
        ## Repeat steps when angles are considered
        sim.runinc(loading, i + 1, stretch_increment, dim, main_file = 'main_angles.in')
        DN_angles = NetworkClass("filler_angles.dat", "test.res")
        cauchy_stress_angles = DN_angles.calculate_stress(dim);
        print("With angles: S_11 = %g, S_22 = %g, S_33 = %g" %tuple(cauchy_stress_angles))
        breakpoint()
    
    
    
    return


if __name__ == "__main__":
    main()