from datetime import datetime
import json
import os
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are



def create_simple_switched_system():
    """
    Create a simple 2D switched linear system
    Two modes: stable and unstable
    """
    # Mode 1: Stable dynamics
    A1 = np.array([
        [-3.0, 7.0],
        [4.0, -6.0]
    ])
    
    # Mode 2: Different dynamics
    A2 = np.array([
        [1.0, -1.0],
        [4.0 , 1.0]
    ])
    
    model = {
        'A': [A1, A2],
    }
    
    return model

def create_simple_switched_systemOSCINSTAB():
    """
    Create a simple 2D switched linear system
    Two modes: stable and unstable
    """
    a1 = 1.0
    a2=2.0
    b1= 2.0
    b2=1.0
    # Mode 1: Stable dynamics
    A1 = np.array([
        [0.1, a1**2],
        [-b1**2, 0.1]
    ])
    
    # Mode 2: Different dynamics
    A2 = np.array([
        [0.1, a2**2],
        [-b2**2, 0.1]
    ])
    
    model = {
        'A': [A1, A2],
    }
    
    return model


#here
if __name__ == "__main__":
    model = create_simple_switched_systemOSCINSTAB()

    for i, A in enumerate(model["A"], start=1):
        eigvals, eigvecs = np.linalg.eig(A)
        singvals = np.linalg.svd(A, compute_uv=False)

        print(f"\nMode {i} - A{i}:")
        print(A)
        print("Autovalori:", eigvals)
        print("Autovettori (colonne):")
        print(eigvecs)
        print("Valori singolari:", singvals)