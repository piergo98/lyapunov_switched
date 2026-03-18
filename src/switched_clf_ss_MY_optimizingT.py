from datetime import datetime
import json
import os
from pathlib import Path
import time
import yaml

import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np



def create_simple_switched_system_SLIDING():
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

def create_simple_switched_system_MARGSTAB():
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
        [0.0, a1**2],
        [-b1**2, 0.0]
    ])
    
    # Mode 2: Different dynamics
    A2 = np.array([
        [0.0, a2**2],
        [-b2**2, 0.0]
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
    
    
    #here, I slightly rotate the ellipses:
    theta = np.radians(30) 

    # 2. Crea la matrice di rotazione R e la sua trasposta
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    R_inv = R.T

    # 3. Calcola la nuova matrice con la dinamica inclinata
    A1 = R @ A1 @ R_inv
    
    model = {
        'A': [A1, A2],
    }
    
    
    
    return model

def create_simple_switched_systemSIMPLIFIED():
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

def build_integrator(f, M_sub, nx):
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
    X0 = ca.MX.sym('X0', nx)
    delta_t = ca.MX.sym('delta_t')
    DT = delta_t / M_sub
    Xk = X0
    Q = 0
    X_sub = ca.MX.zeros(nx, M_sub)
    
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


def solve_switched_clf_optimization(
    switched_model,
    NX=2,
    N=50,
    T=3.0,
    M_RK4=5,
    x0=(2.0, 1.0),
    alpha=0.1,
    eps_cont=1e-5,
    slack_weight=1e4,
    terminal_weight=1e3,
    delta_init=None,
    P_init=None,
):
    """Solve the CLF-based switched optimal control problem and return optimal quantities for plotting.

    Parameters
    ----------
    delta_init : None | float | array-like of shape (N,)
        Initial guess for phase durations Delta. If None, use default heuristic.
    P_init : None | list/tuple of length NM with (NX,NX) arrays
        Initial guess for Lyapunov matrices P per mode. If None, identity-based init is used.
    """
    print("Setting up Switched MPC problem...")

    NM = len(switched_model['A'])
    max_x_variation = 0.2 * M_RK4

    # Compute Lipschitz constants per mode (operator norm).
    K_lipschitz = []
    for i, A in enumerate(switched_model['A']):
        singular_values = np.linalg.svd(A, compute_uv=False)
        K = np.max(singular_values)
        K_lipschitz.append(K)
        print(f"Mode {i}: A =\n{A}")
        print(f"  Singular values: {singular_values}")
        print("  K source: sigma_max(A)")
        print(f"  Lipschitz constant K_{i} = {K:.6f}")
    print()

    x = ca.MX.sym('x', NX)
    Q = ca.diag([1.0, 1.0])
    As = [switched_model['A'][k % NM] for k in range(N)]
    xr = ca.DM([0.0, 0.0])
    stage_cost = (x - xr).T @ Q @ (x - xr)

    f_dyn_list = []
    for i, A in enumerate(As):
        x_dot = A @ x
        f = ca.Function(f'f_dyn_mode_{i}', [x], [x_dot, stage_cost], ['x'], ['x_dot', 'l'])
        f_dyn = build_integrator(f, M_RK4, NX)
        f_dyn_list.append(f_dyn)

    # NLP containers
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # Time interval decision vars
    Delta = ca.MX.sym('Delta', N)
    w += [Delta]

    delta_lb = [0.0] * N
    delta_ub = []
    for k in range(N):
        mode_k = k % NM
        K_k = K_lipschitz[mode_k]
        if K_k > 1e-10:
            delta_ub_k = max_x_variation / K_k
        else:
            delta_ub_k = T
        delta_ub.append(delta_ub_k)

    lbw += delta_lb
    ubw += delta_ub

    if delta_init is None:
        delta_init_vec = np.array([min(T / N, delta_ub[i] * 0.5) for i in range(N)], dtype=float)
    elif np.isscalar(delta_init):
        delta_init_vec = np.full(N, float(delta_init), dtype=float)
    else:
        delta_init_vec = np.array(delta_init, dtype=float).reshape(-1)
        if delta_init_vec.size != N:
            raise ValueError(f"delta_init must have length {N}, got {delta_init_vec.size}")

    delta_init_vec = np.clip(delta_init_vec, np.array(delta_lb), np.array(delta_ub))
    w0 += delta_init_vec.tolist()

    print("Delta bounds based on Lipschitz constants:")
    print(f"  max_x_variation = {max_x_variation}")
    print(f"  M_RK4 = {M_RK4} (used only for integration and sub-step CLF checks)")
    for i in range(NM):
        K_i = K_lipschitz[i]
        if K_i > 1e-10:
            delta_ub_i = max_x_variation / K_i
        else:
            delta_ub_i = T
        print(f"  Mode {i}: sigma_max(A) = {K_i:.6f}, delta_ub = {delta_ub_i:.6f}s")

    # Slack vars for CLF decay
    S_clf = ca.MX.sym('S_clf', N, M_RK4)
    w += [ca.vec(S_clf)]
    lbw += [0.0] * (N * M_RK4)
    ubw += [1e2] * (N * M_RK4)
    w0 += [1e-6] * (N * M_RK4)

    # Lyapunov matrices through Cholesky-like factors
    Xk = ca.DM(x0)
    P_lyap = []
    eps_pd = 1e-3
    tril_size = NX * (NX + 1) // 2
    diag_indices = np.cumsum(np.arange(1, NX + 1)) - 1

    if P_init is not None and len(P_init) != NM:
        raise ValueError(f"P_init must contain {NM} matrices (one per mode), got {len(P_init)}")

    default_L_guess = [1.0 if idx in diag_indices else 0.0 for idx in range(tril_size)]

    for i in range(NM):
        L_entries = ca.MX.sym('L' + str(i), tril_size)
        w += [L_entries]
        lbw += [-20.0] * tril_size
        ubw += [20.0] * tril_size

        if P_init is None:
            L_guess_vec = default_L_guess
        else:
            P_guess = np.array(P_init[i], dtype=float)
            if P_guess.shape != (NX, NX):
                raise ValueError(f"P_init[{i}] must have shape ({NX}, {NX}), got {P_guess.shape}")

            P_guess = 0.5 * (P_guess + P_guess.T)
            P_for_chol = P_guess - eps_pd * np.eye(NX)
            try:
                L_guess = np.linalg.cholesky(P_for_chol)
            except np.linalg.LinAlgError:
                # Fallback: small regularization, then identity if still not SPD
                try:
                    L_guess = np.linalg.cholesky(P_for_chol + 1e-6 * np.eye(NX))
                except np.linalg.LinAlgError:
                    print(f"Warning: invalid P_init[{i}] for Cholesky; using default identity-based init.")
                    L_guess = np.eye(NX)

            L_guess_vec = []
            for row in range(NX):
                for col in range(row + 1):
                    L_guess_vec.append(float(L_guess[row, col]))

        L_guess_vec = np.clip(np.array(L_guess_vec, dtype=float), -20.0, 20.0).tolist()
        w0 += L_guess_vec

        Lk = ca.MX.zeros(NX, NX)
        entry_idx = 0
        for row in range(NX):
            for col in range(row + 1):
                Lk[row, col] = L_entries[entry_idx]
                entry_idx += 1

        Pk = Lk @ Lk.T + eps_pd * ca.DM_eye(NX)
        P_lyap.append(Pk)

    for k in range(N):
        mode_k = k % NM
        if k > 0:
            mode_prev = (k - 1) % NM
            if mode_k != mode_prev:
                V_prev = Xk.T @ P_lyap[mode_prev] @ Xk
                V_curr = Xk.T @ P_lyap[mode_k] @ Xk
                g += [ca.reshape(V_curr - V_prev, -1, 1)]
                lbg += [-1e3]
                ubg += [eps_cont]

        Pk = P_lyap[mode_k]
        Xk_end, l_k, Xk_sub = f_dyn_list[k](Xk, Delta[k])

        decay_factor_sub = ca.exp(-alpha * Delta[k] / M_RK4)
        x_prev_sub = Xk
        for j in range(M_RK4):
            x_curr_sub = Xk_sub[:, j]
            V_prev_sub = x_prev_sub.T @ Pk @ x_prev_sub
            V_curr_sub = x_curr_sub.T @ Pk @ x_curr_sub
            s_kj = S_clf[k, j]
            g += [ca.reshape(V_curr_sub - decay_factor_sub * V_prev_sub - s_kj, -1, 1)]
            lbg += [-1e3]
            ubg += [0.0]
            x_prev_sub = x_curr_sub

        Vk_node = Xk.T @ Pk @ Xk
        Vkp1_node = Xk_end.T @ Pk @ Xk_end
        decay_factor_node = ca.exp(-alpha * Delta[k])
        g += [ca.reshape(Vkp1_node - decay_factor_node * Vk_node, -1, 1)]
        lbg += [-1e3]
        ubg += [0.0]

        Xk = Xk_end
        J += l_k
        J += slack_weight * ca.sum2(S_clf[k, :])

    J += terminal_weight * ca.norm_2(Xk) ** 2

    mode_last = (N - 1) % NM
    mode_first = 0
    if mode_last != mode_first:
        V_last = Xk.T @ P_lyap[mode_last] @ Xk
        V_first = Xk.T @ P_lyap[mode_first] @ Xk
        g += [ca.reshape(V_first - V_last, -1, 1)]
        lbg += [-1e3]
        ubg += [eps_cont]

    g += [ca.reshape(ca.sum1(Delta), -1, 1)]
    lbg += [2.0]
    ubg += [10.0]

    nlp = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
    print("NLP problem created.")
    decision_variables_num = sum([var.size1() * var.size2() for var in w])
    constraints_num = sum([constr.size1() * constr.size2() for constr in g])
    print(f"Number of decision variables: {decision_variables_num}")
    print(f"Number of constraints: {constraints_num}")

    opts = {
        "expand": True,
        "ipopt": {
            "print_level": 5,
            "max_iter": 5000,
            "tol": 1e-6,
            "hsllib": "/usr/local/lib/libcoinhsl.so",
            "linear_solver": "ma27",
        }
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    print("Solving NLP...")
    solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    print("NLP solved.")

    stats = solver.stats()
    ipopt_iter = stats['iter_count'] if 'iter_count' in stats else -1
    solver_success = bool(stats.get('success', False))
    return_status = str(stats.get('return_status', 'unknown'))
    print(f"IPOPT iterations: {ipopt_iter}")
    print(f"Solver success: {solver_success} ({return_status})")

    w_opt = solution['x'].full().flatten()
    Delta_opt = w_opt[:N]

    idx = N
    S_opt = w_opt[idx:idx + N * M_RK4].reshape(N, M_RK4)

    L_opt = []
    P_opt = []
    idx += N * M_RK4
    for _ in range(NM):
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
    print(f"  - {S_opt.size} CLF slacks ({N}x{M_RK4})")
    print(f"  - {len(L_opt)} Lyapunov factors (one per mode)")
    print(f"  - {len(P_opt)} Lyapunov matrices reconstructed from L")
    print(f"Delta_opt shape: {Delta_opt.shape}")
    print(f"S_opt shape: {S_opt.shape}")

    X_nodes = np.zeros((NX, N + 1))
    X_nodes[:, 0] = x0

    V_values = np.zeros(N + 1)
    V_values[0] = x0[0]**2 * P_opt[0][0, 0] + 2 * x0[0] * x0[1] * P_opt[0][0, 1] + x0[1]**2 * P_opt[0][1, 1]
    for k in range(N):
        x_next, _, _ = f_dyn_list[k](X_nodes[:, k], Delta_opt[k])
        X_nodes[:, k + 1] = np.array(x_next).flatten()
        xk1 = X_nodes[:, k + 1]
        mode_k = k % NM
        P_k = P_opt[mode_k]
        V_values[k + 1] = xk1[0]**2 * P_k[0, 0] + 2 * xk1[0] * xk1[1] * P_k[0, 1] + xk1[1]**2 * P_k[1, 1]

    print(f"X_nodes shape: {X_nodes.shape}")

    cumulative_time_nodes = np.zeros(N + 1)
    cumulative_time_nodes[1:] = np.cumsum(Delta_opt)

    print("Optimal trajectories extracted and reconstructed.")

    initial_distance = np.linalg.norm(X_nodes[:, 0])
    final_distance = np.linalg.norm(X_nodes[:, -1])
    max_state_deviation = np.max(np.linalg.norm(X_nodes, axis=0))

    return {
        'solution': solution,
        'solver_success': solver_success,
        'return_status': return_status,
        'ipopt_iter': ipopt_iter,
        'K_lipschitz': K_lipschitz,
        'Delta_opt': Delta_opt,
        'S_opt': S_opt,
        'L_opt': L_opt,
        'P_opt': P_opt,
        'X_nodes': X_nodes,
        'V_values': V_values,
        'cumulative_time_nodes': cumulative_time_nodes,
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'max_state_deviation': max_state_deviation,
        'N': N,
        'NX': NX,
        'NM': NM,
        'M_RK4': M_RK4,
        'T': T,
        'x0': np.array(x0),
        'alpha': alpha,
        'max_x_variation': max_x_variation,
    }

#funzione fatta da copilot per controllare la bontà della soluzione, confrontare due soluzioni e plottare i risultati
def run_solution_sanity_checks(opt_results, label='Solution'):
    """Run basic sanity checks on one optimization result and print a short report."""
    Delta_opt = opt_results['Delta_opt']
    K_lipschitz = opt_results['K_lipschitz']
    P_opt = opt_results['P_opt']
    S_opt = opt_results['S_opt']
    cumulative_time_nodes = opt_results['cumulative_time_nodes']
    max_x_variation = opt_results['max_x_variation']
    solver_success = opt_results.get('solver_success', False)
    return_status = opt_results.get('return_status', 'unknown')

    print(f"\nSanity checks - {label}:")
    print(f"  Solver status: success={solver_success}, return_status={return_status}")

    # Time sum constraint
    sum_delta = float(cumulative_time_nodes[-1])
    sum_ok = (2.0 - 1e-8) <= sum_delta <= (10.0 + 1e-8)
    print(f"  Sum(delta) = {sum_delta:.6f} -> {'OK' if sum_ok else 'VIOLATION'} (expected in [2, 10])")

    # Per-phase delta bound check
    max_violation = 0.0
    for k, d in enumerate(Delta_opt):
        mode_k = k % len(K_lipschitz)
        K_k = K_lipschitz[mode_k]
        if K_k > 1e-10:
            bound = max_x_variation / K_k
        else:
            bound = np.inf
        if np.isfinite(bound):
            max_violation = max(max_violation, float(d - bound))
    print(f"  Max delta bound violation = {max_violation:.3e} -> {'OK' if max_violation <= 1e-8 else 'VIOLATION'}")

    # P positive definiteness check
    min_eig_all = np.inf
    for P in P_opt:
        eigs = np.linalg.eigvalsh(0.5 * (P + P.T))
        min_eig_all = min(min_eig_all, float(np.min(eigs)))
    print(f"  Min eigenvalue among P_k = {min_eig_all:.3e} -> {'OK' if min_eig_all > 0 else 'VIOLATION'}")

    # Slack check
    max_slack = float(np.max(S_opt))
    mean_slack = float(np.mean(S_opt))
    print(f"  Slack stats: max={max_slack:.3e}, mean={mean_slack:.3e}")

#anche questa est iniziativa di copilot  
def compare_two_solutions(opt_results_1, opt_results_2):
    """Compare two solutions to quantify how close they are."""
    d1 = opt_results_1['Delta_opt']
    d2 = opt_results_2['Delta_opt']
    xN1 = opt_results_1['X_nodes'][:, -1]
    xN2 = opt_results_2['X_nodes'][:, -1]
    v1 = opt_results_1['V_values']
    v2 = opt_results_2['V_values']
    p1 = np.array(opt_results_1['P_opt'])
    p2 = np.array(opt_results_2['P_opt'])
    obj_1 = float(opt_results_1['solution']['f'].full()[0, 0])
    obj_2 = float(opt_results_2['solution']['f'].full()[0, 0])

    delta_diff_inf = float(np.max(np.abs(d1 - d2)))
    delta_diff_rel = delta_diff_inf / max(float(np.max(np.abs(d1))), 1e-12)
    xN_diff = float(np.linalg.norm(xN1 - xN2))
    V_diff_inf = float(np.max(np.abs(v1 - v2)))
    P_diff_fro = float(np.linalg.norm(p1 - p2))
    obj_abs = abs(obj_2 - obj_1)
    obj_rel = obj_abs / max(abs(obj_1), 1e-12)

    print("\nNumerical comparison (first vs second solve):")
    print(f"  ||Delta1-Delta2||_inf = {delta_diff_inf:.3e} (rel {delta_diff_rel:.3e})")
    print(f"  ||xN1-xN2||_2        = {xN_diff:.3e}")
    print(f"  ||V1-V2||_inf        = {V_diff_inf:.3e}")
    print(f"  ||P1-P2||_F          = {P_diff_fro:.3e}")
    print(f"  Objective diff       = {obj_abs:.3e} (rel {obj_rel:.3e})")


def plot_solution_results(opt_results, label='Solution'):
    """Plot trajectories and diagnostics for one optimization result."""
    K_lipschitz = opt_results['K_lipschitz']
    Delta_opt = opt_results['Delta_opt']
    X_nodes = opt_results['X_nodes']
    V_values = opt_results['V_values']
    cumulative_time_nodes = opt_results['cumulative_time_nodes']
    N = opt_results['N']
    NM = opt_results['NM']
    T = opt_results['T']
    alpha = opt_results['alpha']
    max_x_variation = opt_results['max_x_variation']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{label} - Switched System with CLF Constraints (N={N}, α={alpha})', fontsize=14, fontweight='bold')

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
    mode_colors = ['lightcoral', 'lightblue']
    mode_labels_added = [False] * NM
    for k in range(N):
        mode_k = k % NM
        seg_label = f'Mode {mode_k}' if not mode_labels_added[mode_k] else None
        plt.plot(
            X_nodes[0, k:k+2],
            X_nodes[1, k:k+2],
            color=mode_colors[mode_k],
            linewidth=2.8,
            alpha=0.9,
            label=seg_label,
            zorder=3,
        )
        mode_labels_added[mode_k] = True

    plt.plot(X_nodes[0, :], X_nodes[1, :], 'ko', markersize=4, alpha=0.8, label='Nodes', zorder=4)
    plt.plot(X_nodes[0, 0], X_nodes[1, 0], 'bs', markersize=14, label='Start', zorder=5, markeredgecolor='darkblue', markeredgewidth=2)
    plt.plot(X_nodes[0, -1], X_nodes[1, -1], 'r*', markersize=16, label='End', zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)
    plt.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, label='Target', zorder=5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.title('Phase Portrait (colored by active mode)', fontweight='bold')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best')
    plt.axis('equal')

    # Time intervals (Delta)
    plt.subplot(2, 3, 4)
    phase_indices = np.arange(N)
    mode_sequence = [k % NM for k in range(N)]
    colors_delta = ['lightcoral' if mode_sequence[k] == 0 else 'lightblue' for k in range(N)]
    plt.bar(phase_indices, Delta_opt, color=colors_delta, alpha=0.7, edgecolor='black', linewidth=0.5, label='Optimal delta')

    # Overlay upper bounds from Lipschitz
    delta_bounds = []
    for k in range(N):
        mode_k = k % NM
        K_k = K_lipschitz[mode_k]
        if K_k > 1e-10:
            delta_bound = max_x_variation / K_k
        else:
            delta_bound = 10.0
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
    plt.plot(cumulative_time_nodes, V_expected, 'r:', linewidth=2, label='V₀·e^(-α·t)', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title('Lyapunov Function Evolution', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('V(x) = x^T P x')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='best')
    plt.yscale('log')

    # Mode switching diagram
    plt.subplot(2, 3, 6)
    mode_sequence = [k % NM for k in range(N)]  # Determine mode for each phase
    colors = ['lightcoral', 'lightblue']
    for k in range(N):
        plt.barh(0, Delta_opt[k], left=cumulative_time_nodes[k],
                 color=colors[mode_sequence[k]], edgecolor='black', linewidth=0.5)
        # Add phase number
        if Delta_opt[k] > 0.02 * T:  # Only label if interval is wide enough
            plt.text(cumulative_time_nodes[k] + Delta_opt[k] / 2, 0, str(k),
                     ha='center', va='center', fontsize=7, fontweight='bold')
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.title('Mode Switching Pattern (Alternating Every Phase)', fontweight='bold')
    plt.xlim([0, cumulative_time_nodes[-1]])
    plt.grid(True, alpha=0.3, axis='x')
    # Add legend for modes
    legend_elements = [Patch(facecolor='lightcoral', edgecolor='black', label='Mode 0'),
                       Patch(facecolor='lightblue', edgecolor='black', label='Mode 1')]
    plt.legend(handles=legend_elements, fontsize=9, loc='upper right')

    plt.tight_layout()


def main():
    # Define system dynamics first
    #switched_model = create_simple_switched_system_SLIDING()   #the sliding mode system from the original paper, with one unstable mode and one stable mode
    switched_model = create_simple_switched_system_MARGSTAB()  #system with marg stab oscillatory modes with different ellipses
    # switched_model = create_simple_switched_systemSIMPLIFIED()
    #switched_model = create_simple_switched_systemOSCINSTAB()

    print("\n=== First solve (default initial guess) ===")
    opt_results_1 = solve_switched_clf_optimization(switched_model)

    print("\n=== Second solve (warm-start from first optimum) ===")
    opt_results_01 = solve_switched_clf_optimization(
        switched_model,
        delta_init=opt_results_1['Delta_opt'],
        P_init=opt_results_1['P_opt'],
    )
    
    opt_results_02 = solve_switched_clf_optimization(
        switched_model,
        delta_init=opt_results_01['Delta_opt'],
        P_init=opt_results_01['P_opt'],
    )
    
    opt_results_2 = solve_switched_clf_optimization(
        switched_model,
        delta_init=opt_results_02['Delta_opt'],
        P_init=opt_results_02['P_opt'],
    )

    obj_1 = float(opt_results_1['solution']['f'].full()[0, 0])
    obj_2 = float(opt_results_2['solution']['f'].full()[0, 0])
    iter_1 = opt_results_1['ipopt_iter']
    iter_2 = opt_results_2['ipopt_iter']
    print("\nWarm-start comparison:")
    print(f"  First solve  - iter: {iter_1}, objective: {obj_1:.6f}")
    print(f"  Second solve - iter: {iter_2}, objective: {obj_2:.6f}")

    run_solution_sanity_checks(opt_results_1, label='First solution')
    run_solution_sanity_checks(opt_results_2, label='Second solution')
    compare_two_solutions(opt_results_1, opt_results_2)

    # Plot both solutions
    plot_solution_results(opt_results_1, label='First solution')
    plot_solution_results(opt_results_2, label='Second solution')

    # Use second solve results for final report
    opt_results = opt_results_2

    solution = opt_results['solution']
    ipopt_iter = opt_results['ipopt_iter']
    K_lipschitz = opt_results['K_lipschitz']
    Delta_opt = opt_results['Delta_opt']
    L_opt = opt_results['L_opt']
    P_opt = opt_results['P_opt']
    X_nodes = opt_results['X_nodes']
    V_values = opt_results['V_values']
    cumulative_time_nodes = opt_results['cumulative_time_nodes']
    initial_distance = opt_results['initial_distance']
    final_distance = opt_results['final_distance']
    max_state_deviation = opt_results['max_state_deviation']
    N = opt_results['N']
    NM = opt_results['NM']
    alpha = opt_results['alpha']
    max_x_variation = opt_results['max_x_variation']
    w_opt = solution['x'].full().flatten()

    print(f"\n{'=' * 60}")
    print("\nOPTIMAL SOLUTION SUMMARY - CLF SINGLE SHOOTING")
    print(f"{'=' * 60}")
    print("Approach: Single shooting with CLF constraints and Lipschitz-based delta bounds")
    print(f"IPOPT Iterations: {ipopt_iter}")
    print(f"Decision variables: {len(w_opt)} total")
    print(f"  - {len(Delta_opt)} time intervals")
    print(f"  - {len(L_opt)} lower-triangular Lyapunov factors (one per mode)")
    print(f"  - {len(P_opt)} Lyapunov matrices reconstructed from L")
    print(f"CLF decay rate (α): {alpha}")
    print("Mode switching: alternating every phase (m=1 fixed)")
    print(f"Total horizon time: {cumulative_time_nodes[-1]:.4f}s (constraint: 1.0 - 10.0s)")
    print(f"Initial state: [{X_nodes[0, 0]:.4f}, {X_nodes[1, 0]:.4f}] (distance: {initial_distance:.4f})")
    print(f"Final state:   [{X_nodes[0, -1]:.4f}, {X_nodes[1, -1]:.4f}] (distance: {final_distance:.4f})")
    print(f"Max state deviation: {max_state_deviation:.4f}")
    print(f"Initial V(x₀): {V_values[0]:.4f}")
    print(f"Final V(x_N):  {V_values[-1]:.4f}")
    print(f"V decay ratio: {V_values[-1] / V_values[0]:.6f}")
    print(f"Objective value: {solution['f'].full()[0, 0]:.6f}")
    print(f"Time intervals - Mean: {np.mean(Delta_opt):.4f}s, Std: {np.std(Delta_opt):.4f}s")
    print(f"Time intervals - Min: {np.min(Delta_opt):.4f}s, Max: {np.max(Delta_opt):.4f}s")

    # Lipschitz-based constraint statistics
    print("\nLipschitz-based Delta Bounds (per mode):")
    print(f"  Max state variation allowed: {max_x_variation}")
    print("  Bound type: node-based (between two NLP nodes)")
    for i in range(NM):
        K_i = K_lipschitz[i]
        if K_i > 1e-10:
            delta_bound = max_x_variation / K_i
            print(f"  Mode {i}: sigma_max(A) = {K_i:.6f}, delta_max = {delta_bound:.6f}s")
        else:
            print(f"  Mode {i}: sigma_max(A) = 0.0, no Lipschitz constraint")

    # Check constraint satisfaction
    print("\nDelta constraint satisfaction (per mode):")
    for i in range(NM):
        K_i = K_lipschitz[i]
        if K_i > 1e-10:
            delta_bound = max_x_variation / K_i
        else:
            delta_bound = np.inf  # No limit for stable modes

        mode_phases = [k for k in range(N) if (k % NM) == i]
        if mode_phases:
            max_delta_mode = np.max(Delta_opt[mode_phases])
            if delta_bound < np.inf:
                margin = delta_bound - max_delta_mode
                print(f"  Mode {i}: delta_bound = {delta_bound:.6f}s, max(delta) = {max_delta_mode:.6f}s, margin = {margin:.6f}s")
                if max_delta_mode > delta_bound * 1.001:
                    print(f"    WARNING: Constraint violated! (max_delta = {max_delta_mode:.6f}s > bound = {delta_bound:.6f}s)")
            else:
                print(f"  Mode {i}: no Lipschitz constraint, max(delta) = {max_delta_mode:.6f}s")

    print(f"\nSum of Delta constraint: 1.0s <= sum(delta) = {cumulative_time_nodes[-1]:.4f}s <= 10.0s")
    print(f"{'=' * 60}\n")

    # Show all generated figures together at the very end
    plt.show()


if __name__ == '__main__':
    main()




