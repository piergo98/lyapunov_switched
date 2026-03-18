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
    a = 1
    b = 2
    lam = 4
    # Mode 1: Stable dynamics
    A1 = np.array([
        [0.0, 1.0],
        [-a+lam, b]
        # [-1.9, 2.0],
        # [-2.0, -0.8]
    ])
    
    # Mode 2: Different dynamics
    A2 = np.array([
        [0.0, 1.0],
        [-a-lam, b]
        # [-2.3, -0.2],
        # [0.0, -2.8]
        # [-1.9, 2.0],
        # [-2.0, -0.8]
    ])
    
    model = {
        'A': [A1, A2],
    }
    
    return model

def instantiate_switched_sequence(model, n_phases, m=1):
    """
    Instantiate a switched sequence of dynamics for the given model and number of phases
    For simplicity, we can alternate between the two modes every m phases
    """
    A_seq = []
    for i in range(n_phases):
        mode_idx = (i // m) % 2  # alternate every m phases
        A_seq.append(model['A'][mode_idx])
    return A_seq

def build_integrator(f, M_sub):
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
    DT = delta_t / M_sub
    Xk = X0
    Q = 0
    X_sub = ca.MX.zeros(NX, M_sub)
    
    for j in range(M_sub):
        k1_x_dot, k1_l = f(Xk)
        k2_x_dot, k2_l = f(Xk + DT/2 * k1_x_dot)
        k3_x_dot, k3_l = f(Xk + DT/2 * k2_x_dot)
        k4_x_dot, k4_l = f(Xk + DT * k3_x_dot)
        Xk = Xk + (DT/6) * (k1_x_dot + 2*k2_x_dot + 2*k3_x_dot + k4_x_dot)
        Q += (DT/6) * (k1_l + 2*k2_l + 2*k3_l + k4_l)
        X_sub[:, j] = Xk

    f_dyn = ca.Function('f_dyn', [X0, delta_t], [Xk, Q, X_sub], ['x', 'delta_t'], ['x_next', 'l', 'x_sub'])
    
    return f_dyn

print("Setting up Switched MPC problem...")

# Define system dynamics first
switched_model = create_simple_switched_system()

# Compute Lipschitz constants: use only unstable eigenvalues (Re(λ) > 0)
# Stable modes don't contribute to state blow-up, so they don't limit delta_k
K_lipschitz = []
for i, A in enumerate(switched_model['A']):
    eigenvalues = np.linalg.eigvals(A)
    # Extract only unstable eigenvalues (real part > 0)
    unstable_eigs = eigenvalues[np.real(eigenvalues) > 1e-10]
    
    if len(unstable_eigs) > 0:
        # Use the maximum real part among unstable eigenvalues
        K = np.max(np.real(unstable_eigs))
    else:
        # All eigenvalues are stable: use the least negative one (closest to 0)
        K = np.max(np.real(eigenvalues))
        if K < 0:
            K = 0.0  # System is stable, no constraint needed
    
    K_lipschitz.append(K)
    print(f"Mode {i}: A =\n{A}")
    print(f"  All eigenvalues: {eigenvalues}")
    print(f"  Unstable eigenvalues: {unstable_eigs if len(unstable_eigs) > 0 else 'None (all stable)'}")
    print(f"  Lipschitz constant K_{i} = {K:.6f}")
print()

# Declare variables
NX = 2      # state dimension
NU = 1      # control dimension
N = 100      # horizon length (number of phases)
T = 3.0     # total time horizon
NM = 2  # number of subsystems (modes)
M_RK4 = 4   # RK4 sub-steps per phase
m=5  # phases per mode: alternate mode at every phase

x_bound = 5.0
max_x_variation = .2  # Maximum allowed variation of x between RK4 nodes

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
As = instantiate_switched_sequence(switched_model, N, m)
xr = ca.DM([0.0, 0.0])  # reference state
ur = ca.DM([0.0])       # reference control input
l = (x - xr).T @ Q @ (x - xr)
f_dyn_list = []
for i in range(len(As)):
    A = As[i]
    x_dot = A @ x
    f = ca.Function('f_dyn', [x], [x_dot, l], ['x'], ['x_dot', 'l'])
    f_dyn = build_integrator(f, M_RK4)
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

x0 = [2.0, 1.0]  # initial state
alpha = 0.1  # CLF decay rate
eps_cont = 1e-5  # continuity tolerance at phase frontiers
slack_weight = 1e4  # penalty on CLF decay slack
terminal_weight = 1e3  # penalty on final state (terminal cost)

# Include time duration variables in the decision variables
Delta = ca.MX.sym('Delta', N)
w += [Delta]

# Compute upper bounds on delta based on Lipschitz constants and max state variation
delta_lb = [0] * N
delta_ub = []
for k in range(N):
    mode_k = (k // m) % NM
    K_k = K_lipschitz[mode_k]
    
    if K_k > 1e-10:
        # Node-based bound: delta_k <= max_x_variation / K_k
        # (no CLF constraints at RK4 internal points)
        delta_ub_k = max_x_variation / K_k
    else:
        # System is stable (K=0), no constraint from Lipschitz
        # Use the original time limit T
        delta_ub_k = T
    
    delta_ub.append(delta_ub_k)
    
lbw += delta_lb
ubw += delta_ub
w0 += [min(T / N, delta_ub[i] * 0.5) for i in range(N)]

print("Delta bounds based on Lipschitz constants:")
print(f"  max_x_variation = {max_x_variation}")
print(f"  M_RK4 (integration only) = {M_RK4}")
for i in range(NM):
    K_i = K_lipschitz[i]
    if K_i > 1e-10:
        delta_ub_i = max_x_variation / K_i
    else:
        delta_ub_i = T
    print(f"  Mode {i}: K = {K_i:.6f}, delta_ub = {delta_ub_i:.6f}s")

# Slack variables for CLF decay at each phase node
S_clf = ca.MX.sym('S_clf', N)
w += [S_clf]
lbw += [0.0] * N
ubw += [1e2] * N
w0 += [1e-6] * N

# Initialize state (convert to CasADi type)
Xk = ca.DM(x0)

# We want to enforce CLF constraints
# a) At phase transitions: V_prev(x_k) = V_curr(x_k) for continuity
# b) At phase nodes: V decays (no RK4 sub-step CLF constraints)

# Define one Lyapunov matrix for each mode through a lower-triangular
# factor L_k so that P_k = L_k L_k^T + eps I.
# This keeps P_k symmetric positive definite by construction.
P_lyap = []
eps_pd = 1e-3
tril_size = NX * (NX + 1) // 2
for i in range(NM):  # One P matrix per mode
    L_entries = ca.MX.sym('L' + str(i), tril_size)
    w += [L_entries]
    lbw += [-20.0] * tril_size
    ubw += [20.0] * tril_size
    w0 += [1.0 if idx in np.cumsum(np.arange(1, NX + 1)) - 1 else 0.0 for idx in range(tril_size)]

    Lk = ca.MX.zeros(NX, NX)
    entry_idx = 0
    for row in range(NX):
        for col in range(row + 1):
            Lk[row, col] = L_entries[entry_idx]
            entry_idx += 1

    Pk = Lk @ Lk.T + eps_pd * ca.DM_eye(NX)
    P_lyap.append(Pk)

# Add continuity constraints at each phase transition within the trajectory loop
    
for k in range(N):
    mode_k = (k // m) % NM
    # Check if this is a mode transition (P_{mode_prev} -> P_{mode_k})
    if k > 0:
        mode_prev = ((k - 1) // m) % NM
        if mode_k != mode_prev:
            # Enforce V continuity at node x_k: x_k^T P_{mode_prev} x_k = x_k^T P_{mode_k} x_k
            V_prev = Xk.T @ P_lyap[mode_prev] @ Xk
            V_curr = Xk.T @ P_lyap[mode_k] @ Xk
            g += [ca.reshape(V_prev - V_curr, -1, 1)]
            lbg += [-eps_cont]
            ubg += [eps_cont]

    Pk = P_lyap[mode_k]
    Xk_end, l_k, _ = f_dyn_list[k](Xk, Delta[k])

    # Enforce continuous-rate CLF decay at phase nodes on the same V_k: x_k -> x_{k+1}
    Vk_node = Xk.T @ Pk @ Xk
    Vkp1_node = Xk_end.T @ Pk @ Xk_end
    decay_factor_node = ca.exp(-alpha * Delta[k])
    g += [ca.reshape(Vkp1_node - decay_factor_node * Vk_node - S_clf[k], -1, 1)]
    lbg += [-1e3]
    ubg += [0.0]
    
    # Forward dynamics
    Xk = Xk_end

    # Accumulate cost
    J += l_k
    J += slack_weight * S_clf[k]

# Add terminal cost: penalizza la norma dello stato finale
J += terminal_weight * ca.norm_2(Xk)**2

# Add periodic CLF continuity at final frontier (k = N)
mode_last = ((N - 1) // m) % NM
mode_first = 0
if mode_last != mode_first:
    V_last = Xk.T @ P_lyap[mode_last] @ Xk
    V_first = Xk.T @ P_lyap[mode_first] @ Xk
    g += [ca.reshape(V_last - V_first, -1, 1)]
    lbg += [-eps_cont]
    ubg += [eps_cont]
    
# Add delta sum constraint: 1.0 <= sum(delta) <= 10.0
g += [ca.reshape(ca.sum1(Delta), -1, 1)]
lbg += [2.0]
ubg += [10.0]
    
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
        "hsllib": "/usr/local/lib/libcoinhsl.so",
        "linear_solver": "ma27",
        # "mu_strategy": "adaptive",
    }
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve the NLP
print("Solving NLP...")
solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print("NLP solved.")

# Extract IPOPT statistics
stats = solver.stats()
ipopt_iter = stats['iter_count'] if 'iter_count' in stats else -1
print(f"IPOPT iterations: {ipopt_iter}")
 
# Extract the optimal decision variables
w_opt = solution['x'].full().flatten()

# Extract Delta (first N elements)
Delta_opt = w_opt[:N]

# Extract CLF slack variables (next N elements)
idx = N
S_opt = w_opt[idx:idx + N]

# Extract Lyapunov factors and reconstruct matrices
L_opt = []
P_opt = []
idx += N
for i in range(NM):
    L_vec = w_opt[idx:idx + tril_size]
    L = np.zeros((NX, NX))
    entry_idx = 0
    for row in range(NX):
        for col in range(row + 1):
            L[row, col] = L_vec[entry_idx]
            entry_idx += 1

    P = L @ L.T + eps_pd * np.eye(NX)
    L_opt.append(L)
    P_opt.append(P)
    idx += tril_size

print(f"Extracted {len(w_opt)} decision variables")
print(f"  - {len(Delta_opt)} time intervals")
print(f"  - {S_opt.size} CLF slacks ({N})")
print(f"  - {len(L_opt)} Lyapunov factors (one per mode)")
print(f"  - {len(P_opt)} Lyapunov matrices reconstructed from L")
print(f"Delta_opt shape: {Delta_opt.shape}")
print(f"S_opt shape: {S_opt.shape}")

# Reconstruct state trajectory by forward simulation with optimal time intervals
X_nodes = np.zeros((NX, N + 1))
X_nodes[:, 0] = x0

# Also compute Lyapunov function values along trajectory
V_values = np.zeros(N + 1)
V_values[0] = x0[0]**2 * P_opt[0][0,0] + 2*x0[0]*x0[1]*P_opt[0][0,1] + x0[1]**2 * P_opt[0][1,1]

for k in range(N):
    # Forward propagate using optimal time interval
    x_next, _, _ = f_dyn_list[k](X_nodes[:, k], Delta_opt[k])
    X_nodes[:, k+1] = np.array(x_next).flatten()
    
    # Compute Lyapunov value using the mode-specific P matrix
    x = X_nodes[:, k+1]
    mode_k = (k // m) % NM
    P_k = P_opt[mode_k]
    V_values[k+1] = x[0]**2 * P_k[0,0] + 2*x[0]*x[1]*P_k[0,1] + x[1]**2 * P_k[1,1]

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
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'Switched System with CLF Constraints (N={N}, α={alpha})', fontsize=14, fontweight='bold')

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
mode_sequence = [(k // m) % 2 for k in range(N)]
colors_delta = ['lightcoral' if mode_sequence[k] == 0 else 'lightblue' for k in range(N)]
plt.bar(phase_indices, Delta_opt, color=colors_delta, alpha=0.7, edgecolor='black', linewidth=0.5, label='Optimal delta')

# Overlay upper bounds from Lipschitz
delta_bounds = []
for k in range(N):
    mode_k = (k // m) % NM
    K_k = K_lipschitz[mode_k]
    if K_k > 1e-10:
        delta_bound = max_x_variation / K_k
    else:
        delta_bound = 10.0  # Large bound for stable modes
    delta_bounds.append(delta_bound)

plt.plot(phase_indices, delta_bounds, 'r--', linewidth=2, label='Lipschitz bound', marker='^', markersize=5)
plt.axhline(y=np.mean(Delta_opt), color='green', linestyle=':', linewidth=2, label=f'Mean delta ({np.mean(Delta_opt):.3f}s)')
plt.title('Time Intervals: Optimal vs Lipschitz Bounds', fontweight='bold')
plt.xlabel('Phase Index')
plt.ylabel('Time (s)')
plt.grid(True, alpha=0.3, axis='y')
plt.legend(fontsize=8)
plt.ylim([0, max(max(Delta_opt), max(delta_bounds)) * 1.2])

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

# Mode switching diagram
plt.subplot(2, 3, 6)
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
plt.title(f'Mode Switching Pattern (Every {m} Phases)', fontweight='bold')
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
print("\nOPTIMAL SOLUTION SUMMARY - CLF SINGLE SHOOTING")
print(f"{'='*60}")
print("Approach: Single shooting with CLF constraints and Lipschitz-based delta bounds")
print(f"IPOPT Iterations: {ipopt_iter}")
print(f"Decision variables: {len(w_opt)} total")
print(f"  - {len(Delta_opt)} time intervals")
print(f"  - {len(L_opt)} lower-triangular Lyapunov factors (one per mode)")
print(f"  - {len(P_opt)} Lyapunov matrices reconstructed from L")
print(f"CLF decay rate (α): {alpha}")
print(f"Mode switching: Every {m} phases")
print(f"Total horizon time: {cumulative_time_nodes[-1]:.4f}s (constraint: 1.0 - 10.0s)")
print(f"Initial state: [{X_nodes[0, 0]:.4f}, {X_nodes[1, 0]:.4f}] (distance: {initial_distance:.4f})")
print(f"Final state:   [{X_nodes[0, -1]:.4f}, {X_nodes[1, -1]:.4f}] (distance: {final_distance:.4f})")
print(f"Max state deviation: {max_state_deviation:.4f}")
print(f"Initial V(x₀): {V_values[0]:.4f}")
print(f"Final V(x_N):  {V_values[-1]:.4f}")
print(f"V decay ratio: {V_values[-1]/V_values[0]:.6f}")
print(f"Objective value: {solution['f'].full()[0, 0]:.6f}")
print(f"Time intervals - Mean: {np.mean(Delta_opt):.4f}s, Std: {np.std(Delta_opt):.4f}s")
print(f"Time intervals - Min: {np.min(Delta_opt):.4f}s, Max: {np.max(Delta_opt):.4f}s")

# Lipschitz-based constraint statistics
print("\nLipschitz-based Delta Bounds (per mode):")
print(f"  Max state variation allowed: {max_x_variation}")
print("  Bound type: node-based (no RK4-point CLF constraints)")
for i in range(NM):
    K_i = K_lipschitz[i]
    if K_i > 1e-10:
        delta_bound = max_x_variation / K_i
        print(f"  Mode {i}: K = {K_i:.6f} (unstable), delta_max = {delta_bound:.6f}s")
    else:
        print(f"  Mode {i}: K = 0.0 (stable), no Lipschitz constraint")

# Check constraint satisfaction
print("\nDelta constraint satisfaction (per mode):")
for i in range(NM):
    K_i = K_lipschitz[i]
    if K_i > 1e-10:
        delta_bound = max_x_variation / K_i
    else:
        delta_bound = np.inf  # No limit for stable modes
    
    mode_phases = [k for k in range(N) if (k // m) % NM == i]
    if mode_phases:
        max_delta_mode = np.max(Delta_opt[mode_phases])
        min_delta_mode = np.min(Delta_opt[mode_phases])
        if delta_bound < np.inf:
            margin = delta_bound - max_delta_mode
            print(f"  Mode {i}: delta_bound = {delta_bound:.6f}s, max(delta) = {max_delta_mode:.6f}s, margin = {margin:.6f}s")
            if max_delta_mode > delta_bound * 1.001:
                print(f"    WARNING: Constraint violated! (max_delta = {max_delta_mode:.6f}s > bound = {delta_bound:.6f}s)")
        else:
            print(f"  Mode {i}: no Lipschitz constraint, max(delta) = {max_delta_mode:.6f}s")

print(f"\nSum of Delta constraint: 1.0s <= sum(delta) = {cumulative_time_nodes[-1]:.4f}s <= 10.0s")
print(f"{'='*60}\n")




