"""
Script containing functions responsible to generate loading informatio and variables
"""
import numpy as np

def create_monotonic_load(max_stretch, increments):
    """
    Create array with monotonic loading history and corresponding increment sizes.
    
    Inputs:
        max_stretch (float): target stretch.
        increments (int): number of increments.
        
    Outputs:
        stretch_array (nparray): array with stretch history.
        stretch_increment (float): stretch increment.
    """
    # Create loading history
    stretch_array = np.linspace(1, max_stretch, increments, endpoint = True)
    
    # Calculate (average) stretch
    stretch_increment = np.mean(np.diff(stretch_array))
    
    return stretch_array, stretch_increment

def deformation_gradient(loading, stretch):
    """
    Compute deformation gradient in the principal space.
    
    Inputs:
        loading (int): identifier of the type of loading (see below):
            1: uniaxial tension
            2: biaxial tension TO DO
            3: pure shear TO DO
            
        stretch (float): scalar charactherising the load "magnitude".
        
    Outputs:
        F (ndarray): 3-row array with the principal stretches
        
    """
    # Initialise
    F = np.zeros(3, )
    
    # Allocate
    if loading == 1:
        F[0] = stretch
        F[1], F[2] = 1 / np.sqrt(stretch), 1 / np.sqrt(stretch)
    
    return F