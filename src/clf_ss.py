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



def create_simple_linear_system():
    """
    Create a simple 2D linear system
    """
    a = 1
    b = 2
    lam = 4
    # Single linear system dynamics
    A = np.array([
        [-1.9, 2.0],
        [-2.0, -0.8]
    ])
    B = np.array([
        [1.0],
        [0.5]
    ])
    
    model = {
        'A': A,
        'B': B
    }
    
    return model

def instantiate_linear_sequence(model, n_phases):
    """
    Instantiate a sequence of the same linear dynamics for the given model and number of phases
    """
    A_seq = [model['A']] * n_phases
    B_seq = [model['B']] * n_phases
    return A_seq, B_seq

def build_integrator(f):
    """Build RK4 integrator for dynamics.
        
    Parameters
    ----------
    f : ca.Function
        Dynamics function returning [x_dot, l].
        
    Returns
    -------
    f_dyn : ca.Function
        Integrated dynamics function returning [x_next, l].
    """
    X0 = ca.MX.sym('X0', NX)
    delta_t = ca.MX.sym('delta_t')
    M = 4
    DT = delta_t / M
    Xk = X0
    Q = 0
    
    for j in range(M):
        k1_x_dot, k1_l = f(Xk)
        k2_x_dot, k2_l = f(Xk + DT/2 * k1_x_dot)
        k3_x_dot, k3_l = f(Xk + DT/2 * k2_x_dot)
        k4_x_dot, k4_l = f(Xk + DT * k3_x_dot)
        Xk = Xk + (DT/6) * (k1_x_dot + 2*k2_x_dot + 2*k3_x_dot + k4_x_dot)
        Q += (DT/6) * (k1_l + 2*k2_l + 2*k3_l + k4_l)

    f_dyn = ca.Function('f_dyn', [X0, delta_t], [Xk, Q], ['x', 'delta_t'], ['x_next', 'l'])
    
    return f_dyn

print("Setting up Linear MPC problem with CLF...")

# Declare variables
NX = 2      # state dimension
NU = 1      # control dimension
N = 100      # horizon length
T = 3.0     # total time horizon

x_bound = 5.0

X_vec = ca.MX.sym('X', NX, N + 1)
Delta_vec = ca.MX.sym('Delta_vec', N)
# vector of all decision variables ordered as [delta, x0, x1, ..., xN]
decvar = ca.veccat(Delta_vec, X_vec)
# to extract trajectories in nice shape from decvar
extract_traj = ca.Function("extract_traj", [decvar], [X_vec, Delta_vec])
traj_to_vec = ca.Function("traj_to_vec", [X_vec, Delta_vec], [decvar])

# Define dynamics and stage cost
x = ca.MX.sym('x', NX)
# u = ca.MX.sym('u', NU)
delta = ca.MX.sym('delta', N)
Q = ca.diag([1.0, 1.0])
linear_model = create_simple_linear_system()
As, _ = instantiate_linear_sequence(linear_model, N)
xr = ca.DM([0.0, 0.0])  # reference state
ur = ca.DM([0.0])       # reference control input
l = (x - xr).T @ Q @ (x - xr)
f_dyn_list = []
for i in range(len(As)):
    A = As[i]
    x_dot = A @ x
    f = ca.Function('f_dyn', [x], [x_dot, l], ['x'], ['x_dot', 'l'])
    f_dyn = build_integrator(f)
    f_dyn_list.append(f_dyn)

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
alpha = 0.5  # CLF decay rate

# Include time duration variables in the decision variables
Delta = ca.MX.sym('Delta', N)
w += [Delta]
lbw += [T/N] * N
ubw += [T/N] * N
w0 += [T / N] * N
# w0 += [T / N * 1.5 ** (i / N) for i in range(N)]

# Initialize state (convert to CasADi type)
Xk = ca.DM(x0)

# We want to enforce CLF constraints
# Enforce: V(x_k+1) - V(x_k) <= -alpha * V(x_k) for stability

# Define a single Lyapunov matrix as a decision variable
Lk = ca.MX.sym('L', NX, NX)
w += [ca.vec(Lk)]
lbw += [-1e3] * NX * NX  # Assuming positive definite matrix
ubw += [1e3] * NX * NX  # Upper bound for Lyapunov matrix elements
w0 += [1.0] * NX * NX  # Initial guess for Lyapunov matrix

# Enforce lower triangular structure
Lk_lower = ca.tril(Lk)

# Construct P
Pk = Lk_lower @ Lk_lower.T  # Automatically positive semi-definite

# To ensure strict positive definiteness, add small regularization to diagonal
eps = 1e-6
P_lyap = Lk_lower @ Lk_lower.T + eps * ca.MX.eye(NX)
    
for k in range(N):
    # Enforce CLF decay constraint
    Vk = Xk.T @ P_lyap @ Xk
    
    Xk_end, _ = f_dyn_list[k](Xk, Delta[k])
    
    Vk_end = Xk_end.T @ P_lyap @ Xk_end
    g += [ca.reshape(Vk_end - Vk + alpha * Vk, -1, 1)]  # Ensure column vector
    lbg += [-1e3]  # Lower bound (can be adjusted)
    ubg += [0.0]   # Upper bound (constraint is <= 0)
    
    # Forward dynamics
    Xk, l_k = f_dyn_list[k](Xk, Delta[k])

    # Accumulate cost
    J += l_k
    
    # Add dynamics constraint (ensure column vector)
    g += [ca.reshape(Xk, -1, 1)]
    lbg += [-x_bound] * NX
    ubg += [x_bound] * NX
    
# Add delta constraint (scalar)
g += [ca.reshape(ca.sum1(Delta) - T, -1, 1)]
lbg += [0.0]
ubg += [0.0]
    
# Add final constraint to reach the target state
# g += [Xk_end - xr]
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
        "tol": 1e-6, 
        "hsllib": "/home/pietro/ThirdParty-HSL/coinhsl-2024.05.15/install/lib/x86_64-linux-gnu/libcoinhsl.so",
        "linear_solver": "ma27",
        # "mu_strategy": "adaptive",
    }
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve the NLP
print("Solving NLP...")
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print("NLP solved.")
 
# Extract the optimal decision variables
w_opt = solution['x'].full().flatten()

# Extract Delta (first N elements)
Delta_opt = w_opt[:N]

# Extract Lyapunov matrix (remaining elements, single matrix of size NX*NX)
idx = N
L_vec = w_opt[idx:idx + NX*NX]
L = L_vec.reshape(NX, NX)
L_lower = np.tril(L)
P_opt = L_lower @ L_lower.T + 1e-6 * np.eye(NX)

print(f"Extracted {len(w_opt)} decision variables")
print(f"  - {len(Delta_opt)} time intervals")
print(f"  - 1 Lyapunov matrix")
print(f"Delta_opt shape: {Delta_opt.shape}")

# Reconstruct state trajectory by forward simulation with optimal time intervals
X_nodes = np.zeros((NX, N + 1))
X_nodes[:, 0] = x0

# Also compute Lyapunov function values along trajectory
V_values = np.zeros(N + 1)
V_values[0] = x0[0]**2 * P_opt[0,0] + 2*x0[0]*x0[1]*P_opt[0,1] + x0[1]**2 * P_opt[1,1]

for k in range(N):
    # Forward propagate using optimal time interval
    x_next, _ = f_dyn_list[k](X_nodes[:, k], Delta_opt[k])
    X_nodes[:, k+1] = np.array(x_next).flatten()
    
    # Compute Lyapunov value using the single P matrix
    x = X_nodes[:, k+1]
    V_values[k+1] = x[0]**2 * P_opt[0,0] + 2*x[0]*x[1]*P_opt[0,1] + x[1]**2 * P_opt[1,1]

print(f"X_nodes shape: {X_nodes.shape}")

# Compute cumulative time for nodes
cumulative_time_nodes = np.zeros(N + 1)
cumulative_time_nodes[1:] = np.cumsum(Delta_opt)

print("Optimal trajectories extracted and reconstructed.")

# Compute summary statistics
initial_distance = np.linalg.norm(X_nodes[:, 0])
final_distance = np.linalg.norm(X_nodes[:, -1])
max_state_deviation = np.max(np.linalg.norm(X_nodes, axis=0))

# Plot trajectories
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f'Linear System with CLF Constraints (N={N}, T={T}s, α={alpha})', fontsize=14, fontweight='bold')

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

# States vs continuous time
plt.subplot(2, 3, 2)
plt.plot(cumulative_time_nodes, X_nodes[0, :], 'b-', linewidth=2, marker='o', markersize=6, label='x₁')
plt.plot(cumulative_time_nodes, X_nodes[1, :], 'r-', linewidth=2, marker='s', markersize=6, label='x₂')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('State Trajectories vs Time', fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')

# Phase portrait
plt.subplot(2, 3, 3)
plt.plot(X_nodes[0, :], X_nodes[1, :], 'g-', linewidth=2.5, alpha=0.7, label='Trajectory')
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
plt.bar(phase_indices, Delta_opt, color='lightblue', alpha=0.7, edgecolor='black', linewidth=0.5)
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

# Lyapunov function evolution
plt.subplot(2, 3, 5)
plt.plot(cumulative_time_nodes, V_values, 'purple', linewidth=2.5, marker='o', markersize=7, label='V(x)')
# Also plot state norm for comparison
state_norms = np.linalg.norm(X_nodes, axis=0)
plt.plot(cumulative_time_nodes, state_norms**2, 'g--', linewidth=2, marker='s', markersize=5, alpha=0.7, label='||x||²')
# Plot expected decay envelope
V_expected = V_values[0] * np.exp(-alpha * cumulative_time_nodes)
plt.plot(cumulative_time_nodes, V_expected, 'r:', linewidth=2, label=f'V₀·e^(-α·t)', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Lyapunov Function Evolution', fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('V(x) = x^T P x')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, loc='best')
plt.yscale('log')

# Distance to origin over time
plt.subplot(2, 3, 6)
distance_to_origin = np.linalg.norm(X_nodes, axis=0)
plt.plot(cumulative_time_nodes, distance_to_origin, 'b-', linewidth=2.5, marker='o', markersize=6, label='||x||')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Distance to Origin', fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('||x||')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9, loc='best')
plt.yscale('log')

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print(f"OPTIMAL SOLUTION SUMMARY - CLF SINGLE SHOOTING")
print(f"{'='*60}")
print(f"Approach: Single shooting with CLF constraints (Linear System)")
print(f"Decision variables: {len(w_opt)} total")
print(f"  - {len(Delta_opt)} time intervals")
print(f"  - 1 Lyapunov matrix")
print(f"CLF decay rate (α): {alpha}")
print(f"Total time: {cumulative_time_nodes[-1]:.4f}s (target: {T:.4f}s)")
print(f"Initial state: [{X_nodes[0, 0]:.4f}, {X_nodes[1, 0]:.4f}] (distance: {initial_distance:.4f})")
print(f"Final state:   [{X_nodes[0, -1]:.4f}, {X_nodes[1, -1]:.4f}] (distance: {final_distance:.4f})")
print(f"Max state deviation: {max_state_deviation:.4f}")
print(f"Initial V(x₀): {V_values[0]:.4f}")
print(f"Final V(x_N):  {V_values[-1]:.4f}")
print(f"V decay ratio: {V_values[-1]/V_values[0]:.6f}")
print(f"Objective value: {solution['f'].full()[0, 0]:.6f}")
print(f"Time intervals - Mean: {np.mean(Delta_opt):.4f}s, Std: {np.std(Delta_opt):.4f}s")
print(f"Time intervals - Min: {np.min(Delta_opt):.4f}s, Max: {np.max(Delta_opt):.4f}s")
print(f"\nOptimal Lyapunov matrix P:")
print(P_opt)
print(f"{'='*60}\n")





