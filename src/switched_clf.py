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
        [0.9, 0.1],
        [-0.1, 0.8]
    ])
    B1 = np.array([
        [1.0],
        [0.5]
    ])
    
    # Mode 2: Different dynamics
    A2 = np.array([
        [2.0, -0.2],
        [0.0, 1.5]
    ])
    B2 = np.array([
        [0.5],
        [1.0]
    ])
    
    model = {
        'A': [A1, A2],
        'B': [B1, B2]
    }
    
    return model

def instantiate_switched_sequence(model, n_phases, m=1):
    """
    Instantiate a switched sequence of dynamics for the given model and number of phases
    For simplicity, we can alternate between the two modes every m phases
    """
    A_seq = []
    B_seq = []
    for i in range(n_phases):
        mode_idx = (i // m) % 2  # alternate every m phases
        A_seq.append(model['A'][mode_idx])
        B_seq.append(model['B'][mode_idx])
    return A_seq, B_seq

def get_collocation_coefficients(d=3):
    """Compute collocation coefficients for a given degree d.
    
    Parameters
    ----------
    d : int
        Degree of collocation polynomials.
        
    Returns
    -------
    C : np.ndarray
        Coefficients of the collocation equation.
    D : np.ndarray
        Coefficients of the continuity equation.
    B : np.ndarray
        Coefficients of the quadrature function.
    """
    # Get collocation points
    tau_root = np.append(0, ca.collocation_points(d, 'legendre'))
    
    # Coefficients of the collocation equation
    C = np.zeros((d + 1, d + 1))
    
    # Coefficients of the continuity equation
    D = np.zeros(d + 1)
    
    # Coefficients of the quadrature function
    B = np.zeros(d + 1)
    
    # Construct polynomial basis
    for j in range(d + 1):
        # Construct Lagrange polynomials
        p = np.poly1d([1])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
        
        # Evaluate the polynomial at the final time
        D[j] = p(1.0)
        
        # Evaluate the time derivative of the polynomial at all collocation points
        pder = np.polyder(p)
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])
        
        # Evaluate the integral of the polynomial
        pint = np.polyint(p)
        B[j] = pint(1.0)
    
    return C, D, B

print("Setting up Switched MPC problem...")

# Declare variables
NX = 2      # state dimension
NU = 1      # control dimension
N = 20      # horizon length
D = 3       # collocation degree
T = 2.0     # total time horizon

x_bound = 5.0

# Define dynamics and stage cost
x = ca.SX.sym('x', NX)
# u = ca.SX.sym('u', NU)
delta = ca.SX.sym('sigma', N)
Q = ca.diag([10.0, 10.0])
switched_model = create_simple_switched_system()
As, _ = instantiate_switched_sequence(switched_model, N, m=2)
C_coeff, D_coeff, B_coeff = get_collocation_coefficients(D)
xr = ca.DM([0.0, 0.0])  # reference state
ur = ca.DM([0.0])       # reference control input
l = (x - xr).T @ Q @ (x - xr)
dyn_list = []
for i in range(len(As)):
    A = As[i]
    x_dot = A @ x
    f_dyn = ca.Function('f_dyn', [x], [x_dot, l], ['x'], ['x_dot', 'l'])
    dyn_list.append(f_dyn)

# No symbolic extraction function needed - we'll extract manually from w_opt

# Create an NLP to minimize the cost over a horizon
w = []      # decision variables
w0 = []     # initial guess
lbw = []    # lower bounds
ubw = []    # upper bounds
J = 0       # objective
g = []      # constraints
lbg = []    # lower bounds on constraints
ubg = []    # upper bounds on constraints
delta_sum = 0  # to accumulate total time

x0 = [2.0, -1.0]  # initial state

# Include time duration variables in the decision variables
Delta = ca.SX.sym('Delta', N)
w += [Delta]
lbw += [0] * N
ubw += [T] * N
w0 += [T] * N

# Initialize state
Xk = ca.SX.sym('X_0', NX)
w += [Xk]
lbw += x0
ubw += x0
w0 += x0
for k in range(N):
    
    # State at collocation points
    Xc = []
    for j in range(D):
        Xkj = ca.SX.sym('X_' + str(k) + '_' + str(j), NX)
        Xc.append(Xkj)
        w += [Xkj]
        lbw += [-x_bound] * NX
        ubw += [x_bound] * NX
        w0 += [0.0] * NX
    
    # Loop over collocation points
    Xk_end = D_coeff[0] * Xk
    id = k % len(As)  # determine which mode we are in based on time step
    for j in range(1, D + 1):
        # Expression for the state derivative at the collocation point
        xp = C_coeff[0, j] * Xk
        for r in range(D):
            xp = xp + C_coeff[r + 1, j] * Xc[r]
        
        # Append collocation equations
        fj, qj = dyn_list[id](Xc[j - 1])
        g += [Delta[k] * fj - xp]
        lbg += [0.0] * NX
        ubg += [0.0] * NX
        
        # Add contribution to the end state
        Xk_end = Xk_end + D_coeff[j] * Xc[j - 1]
        
        # Add contribution to quadrature function
        J += B_coeff[j] * qj * Delta[k]
    
    # Next state
    Xk = ca.SX.sym('X_' + str(k + 1), NX)
    w += [Xk]
    lbw += [-x_bound] * NX
    ubw += [x_bound] * NX
    w0 += [0.0] * NX
    
    # Add dynamics constraint
    g += [Xk_end - Xk]
    lbg += [0.0] * NX
    ubg += [0.0] * NX
    
# Add delta constraint
g += [ca.sum1(Delta) - T]
lbg += [0.0]
ubg += [0.0]
    
# Add final constraint to reach the target state
# g += [Xk - xr]
# lbg += [0.0] * NX
# ubg += [0.0] * NX
# E = solve_discrete_are(A.full(), B.full(), Q.full(), R.full())
# J += (Xk - xr).T @ ca.DM(E) @ (Xk - xr)

# Create NLP solver
nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
print("NLP problem created.")
decision_variables_num = sum([var.size1()*var.size2() for var in w])
constraints_num = sum([constr.size1()*constr.size2() for constr in g])
print(f"Number of decision variables: {decision_variables_num}")
print(f"Number of constraints: {constraints_num}")

# Create the NLP solver
opts = {
    "expand": True, 
    "ipopt": {
        "print_level": 5, 
        "max_iter": 5000, 
        "tol": 1e-8, 
        "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
        "linear_solver": "ma27",
    }
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve the NLP
print("Solving NLP...")
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print("NLP solved.")
 
# Extract the optimal trajectories
w_opt = solution['x'].full().flatten()

# Manually extract states, delta, and collocation points from w_opt
# Structure: [Delta (N), X_0, X_0_0, X_0_1, X_0_2, X_1, X_1_0, X_1_1, X_1_2, X_2, ...]
X_nodes = np.zeros((NX, N + 1))  # States at nodes
Delta_opt = np.zeros(N)          # Time intervals
X_coll = np.zeros((N, D, NX))    # Collocation points

idx = 0

# Extract all Delta values first
Delta_opt = w_opt[idx:idx+N]
idx += N

# Extract initial state X_0
X_nodes[:, 0] = w_opt[idx:idx+NX]
idx += NX

# Extract for each interval
for k in range(N):
    # Extract collocation points X_k_0, X_k_1, X_k_2
    for j in range(D):
        X_coll[k, j, :] = w_opt[idx:idx+NX]
        idx += NX
    
    # Extract next state X_{k+1}
    X_nodes[:, k+1] = w_opt[idx:idx+NX]
    idx += NX

print(f"Extracted {idx} decision variables")

# Compute cumulative time for nodes
cumulative_time_nodes = np.zeros(N + 1)
cumulative_time_nodes[1:] = np.cumsum(Delta_opt)

# Get collocation time points within each interval
tau_root = np.append(0, ca.collocation_points(D, 'legendre'))

# Compute times and states for collocation points (for smoother plotting)
n_total_points = N + 1 + N * D  # nodes + all collocation points
times_all = []
states_all = [[], []]  # for x1 and x2

# Add initial state
times_all.append(0.0)
states_all[0].append(X_nodes[0, 0])
states_all[1].append(X_nodes[1, 0])

for k in range(N):
    t_start = cumulative_time_nodes[k]
    delta_k = Delta_opt[k]
    
    # Add collocation points within this interval
    for j in range(D):
        t_coll = t_start + tau_root[j+1] * delta_k  # tau_root[0] is 0, skip it
        times_all.append(t_coll)
        states_all[0].append(X_coll[k, j, 0])
        states_all[1].append(X_coll[k, j, 1])
    
    # Add node at end of interval
    times_all.append(cumulative_time_nodes[k+1])
    states_all[0].append(X_nodes[0, k+1])
    states_all[1].append(X_nodes[1, k+1])

times_all = np.array(times_all)
states_all[0] = np.array(states_all[0])
states_all[1] = np.array(states_all[1])

print("Optimal trajectories extracted.")

# Compute summary statistics
initial_distance = np.linalg.norm(X_nodes[:, 0])
final_distance = np.linalg.norm(X_nodes[:, -1])
max_state_deviation = np.max(np.linalg.norm(X_nodes, axis=0))

# Plot trajectories
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f'Switched System MPC (N={N}, D={D}, T={T}s)', fontsize=14, fontweight='bold')

# States vs time step (nodes only)
plt.subplot(2, 3, 1)
time_steps = np.arange(N + 1)
plt.plot(time_steps, X_nodes[0, :], 'b-', linewidth=2, label='x₁', marker='o', markersize=6)
plt.plot(time_steps, X_nodes[1, :], 'r-', linewidth=2, label='x₂', marker='s', markersize=6)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('State Trajectories at Nodes', fontweight='bold')
plt.xlabel('Node Index')
plt.ylabel('States')
plt.grid(True, alpha=0.3)
plt.legend(loc='best')

# States vs continuous time (with collocation points)
plt.subplot(2, 3, 2)
plt.plot(times_all, states_all[0], 'b-', linewidth=1.5, alpha=0.7)
plt.plot(times_all, states_all[1], 'r-', linewidth=1.5, alpha=0.7)
# Mark nodes with larger markers
plt.plot(cumulative_time_nodes, X_nodes[0, :], 'bo', markersize=8, label='x₁ nodes')
plt.plot(cumulative_time_nodes, X_nodes[1, :], 'rs', markersize=8, label='x₂ nodes')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('State Trajectories vs Time (with Collocation)', fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')

# Phase portrait (with collocation points)
plt.subplot(2, 3, 3)
plt.plot(states_all[0], states_all[1], 'g-', linewidth=1.5, alpha=0.6, label='Trajectory')
plt.plot(X_nodes[0, :], X_nodes[1, :], 'go', markersize=7, alpha=0.8, label='Nodes', zorder=4)
plt.plot(X_nodes[0, 0], X_nodes[1, 0], 'bs', markersize=14, label='Start', zorder=5, markeredgecolor='darkblue', markeredgewidth=2)
plt.plot(X_nodes[0, -1], X_nodes[1, -1], 'r*', markersize=16, label='End', zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)
plt.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, label='Target', zorder=5)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.title('Phase Portrait', fontweight='bold')
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')
plt.axis('equal')

# Time intervals (Delta)
plt.subplot(2, 3, 4)
phase_indices = np.arange(N)
m = 2  # phases per mode
mode_sequence = [(k // m) % 2 for k in range(N)]
colors_delta = ['lightcoral' if mode_sequence[k] == 0 else 'lightblue' for k in range(N)]
plt.bar(phase_indices, Delta_opt, color=colors_delta, alpha=0.7, edgecolor='black', linewidth=0.5)
plt.axhline(y=T/N, color='darkred', linestyle='--', linewidth=2, label=f'Uniform ({T/N:.3f}s)')
mean_delta = np.mean(Delta_opt)
std_delta = np.std(Delta_opt)
plt.axhline(y=mean_delta, color='green', linestyle=':', linewidth=2, label=f'Mean ({mean_delta:.3f}s)')
plt.title('Time Interval per Phase', fontweight='bold')
plt.xlabel('Phase Index')
plt.ylabel('Time Interval (s)')
plt.grid(True, alpha=0.3, axis='y')
plt.legend(fontsize=8)
plt.ylim([0, max(Delta_opt) * 1.15])

# Collocation points visualization for first few intervals
plt.subplot(2, 3, 5)
n_show = min(5, N)  # Show first 5 intervals
for k in range(n_show):
    t_start = cumulative_time_nodes[k]
    delta_k = Delta_opt[k]
    
    # Plot interval span
    m = 2
    mode_k = (k // m) % 2
    color_k = 'lightcoral' if mode_k == 0 else 'lightblue'
    plt.axvspan(t_start, t_start + delta_k, alpha=0.2, color=color_k)
    
    # Plot node at start
    plt.plot(t_start, X_nodes[0, k], 'o', markersize=10, color=f'C{k}', 
             markeredgecolor='black', markeredgewidth=1, label=f'Phase {k}' if k < 3 else None)
    
    # Plot collocation points
    for j in range(D):
        t_coll = t_start + tau_root[j+1] * delta_k
        plt.plot(t_coll, X_coll[k, j, 0], 'x', markersize=8, markeredgewidth=2, color=f'C{k}', alpha=0.8)
    
    # Plot node at end
    if k == n_show - 1:
        plt.plot(t_start + delta_k, X_nodes[0, k+1], 'o', markersize=10, color=f'C{k}', 
                 markeredgecolor='black', markeredgewidth=1)

plt.title(f'x₁: Nodes (●) & Collocation (×) - First {n_show} Phases', fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('x₁ State')
plt.grid(True, alpha=0.3)
if n_show <= 3:
    plt.legend(fontsize=8, loc='best')

# Mode switching diagram
plt.subplot(2, 3, 6)
m = 2  # phases per mode (must match the value in instantiate_switched_sequence)
mode_sequence = [(k // m) % 2 for k in range(N)]  # Determine mode for each phase
colors = ['lightcoral', 'lightblue']
for k in range(N):
    plt.barh(0, Delta_opt[k], left=cumulative_time_nodes[k], 
             color=colors[mode_sequence[k]], edgecolor='black', linewidth=0.5)
    # Add phase number
    if Delta_opt[k] > 0.02 * T:  # Only label if interval is wide enough
        plt.text(cumulative_time_nodes[k] + Delta_opt[k]/2, 0, str(k), 
                ha='center', va='center', fontsize=7, fontweight='bold')
plt.yticks([])
plt.xlabel('Time (s)')
plt.title('Mode Switching Pattern', fontweight='bold')
plt.xlim([0, cumulative_time_nodes[-1]])
plt.grid(True, alpha=0.3, axis='x')
# Add legend for modes
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightcoral', edgecolor='black', label='Mode 0'),
                   Patch(facecolor='lightblue', edgecolor='black', label='Mode 1')]
plt.legend(handles=legend_elements, fontsize=9, loc='upper right')

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print(f"OPTIMAL SOLUTION SUMMARY")
print(f"{'='*60}")
print(f"Total time: {cumulative_time_nodes[-1]:.4f}s (target: {T:.4f}s)")
print(f"Initial state: [{X_nodes[0, 0]:.4f}, {X_nodes[1, 0]:.4f}] (distance: {initial_distance:.4f})")
print(f"Final state:   [{X_nodes[0, -1]:.4f}, {X_nodes[1, -1]:.4f}] (distance: {final_distance:.4f})")
print(f"Max state deviation: {max_state_deviation:.4f}")
print(f"Objective value: {solution['f'].full()[0, 0]:.6f}")
print(f"Time intervals - Mean: {np.mean(Delta_opt):.4f}s, Std: {np.std(Delta_opt):.4f}s")
print(f"Time intervals - Min: {np.min(Delta_opt):.4f}s, Max: {np.max(Delta_opt):.4f}s")
print(f"{'='*60}\n")





