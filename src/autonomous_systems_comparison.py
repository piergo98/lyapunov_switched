"""
Forward simulation of multiple autonomous linear systems from the same initial condition.
Compares the behavior of different system dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Patch


def simulate_autonomous_system(A, x0, t_span, t_eval):
    """
    Simulate autonomous linear system: dx/dt = Ax
    
    Parameters
    ----------
    A : np.ndarray
        System matrix (n x n)
    x0 : np.ndarray
        Initial condition (n,)
    t_span : tuple
        Time span (t_start, t_end)
    t_eval : np.ndarray
        Time points to evaluate solution
        
    Returns
    -------
    solution : OdeResult
        Solution object with time and state trajectories
    """
    def dynamics(t, x):
        return A @ x
    
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
    return sol


def plot_vector_field(ax, A, x_range, y_range, n_points=15, color='gray', alpha=0.4, scale=None):
    """
    Plot the vector field for the system dx/dt = Ax
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    A : np.ndarray
        System matrix
    x_range : tuple
        Range for x1 (min, max)
    y_range : tuple
        Range for x2 (min, max)
    n_points : int
        Number of grid points in each direction
    color : str
        Arrow color
    alpha : float
        Arrow transparency
    scale : float or None
        Scaling factor for arrows
    """
    x1 = np.linspace(x_range[0], x_range[1], n_points)
    x2 = np.linspace(y_range[0], y_range[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Compute vector field
    DX1 = A[0, 0] * X1 + A[0, 1] * X2
    DX2 = A[1, 0] * X1 + A[1, 1] * X2
    
    # Normalize vectors for better visualization
    magnitude = np.sqrt(DX1**2 + DX2**2)
    # Avoid division by zero
    magnitude = np.where(magnitude == 0, 1, magnitude)
    
    # Plot vector field
    ax.quiver(X1, X2, DX1/magnitude, DX2/magnitude, 
             magnitude, cmap='viridis', alpha=alpha, scale=scale, width=0.003,
             headwidth=4, headlength=5, headaxislength=4.5)


def main():
    # Common initial condition
    x0 = np.array([2.0, -1.0])
    
    # Time settings
    t_start = 0.0
    t_end = 2.0
    t_eval = np.linspace(t_start, t_end, 500)
    
    # Define multiple autonomous systems
    a = 1
    b = 2
    lam = 4
    systems = {
        'System 1': np.array([
            [0.0, 1.0],
            [-a+lam, b]
            # [-1.9, 2.0],
            # [-2.0, -0.8]
        ]),
        'System 2 ': np.array([
            [0.0, 1.0],
            [-a-lam, b]
            # [-2.3, -0.2],
            # [0.0, -2.8]
        ]),
    }
    
    # Simulate each system
    solutions = {}
    colors = ['blue', 'red', 'green', 'purple']
    
    print("="*70)
    print("AUTONOMOUS LINEAR SYSTEMS COMPARISON")
    print("="*70)
    print(f"Initial condition: x0 = [{x0[0]:.3f}, {x0[1]:.3f}]")
    print(f"Time span: [{t_start:.1f}, {t_end:.1f}]")
    print()
    
    for (name, A), color in zip(systems.items(), colors):
        sol = simulate_autonomous_system(A, x0, (t_start, t_end), t_eval)
        solutions[name] = {'sol': sol, 'A': A, 'color': color}
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        
        # Compute final state and distance
        x_final = sol.y[:, -1]
        initial_distance = np.linalg.norm(x0)
        final_distance = np.linalg.norm(x_final)
        
        print(f"{name}:")
        print(f"  Matrix A:")
        print(f"    {A[0]}")
        print(f"    {A[1]}")
        print(f"  Eigenvalues: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")
        print(f"  Final state: [{x_final[0]:.4f}, {x_final[1]:.4f}]")
        print(f"  Distance - Initial: {initial_distance:.4f}, Final: {final_distance:.4f}")
        print()
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Autonomous Linear Systems Comparison', fontsize=16, fontweight='bold')
    
    # Determine plot ranges based on trajectories
    all_x1 = np.concatenate([sol['sol'].y[0, :] for sol in solutions.values()])
    all_x2 = np.concatenate([sol['sol'].y[1, :] for sol in solutions.values()])
    x1_range = (np.min(all_x1) * 1.2, np.max(all_x1) * 1.2)
    x2_range = (np.min(all_x2) * 1.2, np.max(all_x2) * 1.2)
    
    # 1. Phase portrait - all systems with combined vector field
    ax1 = plt.subplot(2, 3, 1)
    for name, data in solutions.items():
        sol = data['sol']
        color = data['color']
        ax1.plot(sol.y[0, :], sol.y[1, :], color=color, linewidth=2, alpha=0.7, label=name.split('(')[0].strip())
        # Mark final point
        ax1.plot(sol.y[0, -1], sol.y[1, -1], 'o', color=color, markersize=8, 
                markeredgecolor='black', markeredgewidth=1)
    
    # Mark initial condition
    ax1.plot(x0[0], x0[1], 's', color='black', markersize=12, 
            label='x₀', zorder=10, markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, label='Origin', zorder=10)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('x₂', fontsize=11)
    ax1.set_title('Phase Portrait', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    ax1.axis('equal')
    
    # 2. State x₁ vs time
    ax2 = plt.subplot(2, 3, 2)
    for name, data in solutions.items():
        sol = data['sol']
        color = data['color']
        ax2.plot(sol.t, sol.y[0, :], color=color, linewidth=2, label=name.split('(')[0].strip())
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('x₁', fontsize=11)
    ax2.set_title('State x₁ vs Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')
    
    # 3. State x₂ vs time
    ax3 = plt.subplot(2, 3, 3)
    for name, data in solutions.items():
        sol = data['sol']
        color = data['color']
        ax3.plot(sol.t, sol.y[1, :], color=color, linewidth=2, label=name.split('(')[0].strip())
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('x₂', fontsize=11)
    ax3.set_title('State x₂ vs Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc='best')
    
    # 4. Distance from origin vs time
    ax4 = plt.subplot(2, 3, 4)
    for name, data in solutions.items():
        sol = data['sol']
        color = data['color']
        distances = np.linalg.norm(sol.y, axis=0)
        ax4.plot(sol.t, distances, color=color, linewidth=2, label=name.split('(')[0].strip())
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('||x||₂', fontsize=11)
    ax4.set_title('Distance from Origin', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8, loc='best')
    ax4.set_yscale('log')
    
    # 5. Individual phase portraits (zoomed) with vector fields
    ax5 = plt.subplot(2, 3, 5)
    for idx, (name, data) in enumerate(solutions.items()):
        sol = data['sol']
        color = data['color']
        # Plot with thinner lines
        ax5.plot(sol.y[0, :], sol.y[1, :], color=color, linewidth=2.5, alpha=0.8, label=name, zorder=5)
        
        # Add arrows to show direction
        n_arrows = 5
        arrow_indices = np.linspace(0, len(sol.t)-1, n_arrows, dtype=int)[:-1]
        for i in arrow_indices:
            dx = sol.y[0, i+1] - sol.y[0, i]
            dy = sol.y[1, i+1] - sol.y[1, i]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 1e-6:
                ax5.arrow(sol.y[0, i], sol.y[1, i], dx*0.8, dy*0.8, 
                         head_width=0.15, head_length=0.12, fc=color, ec=color, alpha=0.7, zorder=6)
    
    ax5.plot(x0[0], x0[1], 's', color='black', markersize=12, 
            label='x₀', zorder=10, markeredgecolor='white', markeredgewidth=1.5)
    ax5.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, zorder=10)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax5.set_xlabel('x₁', fontsize=11)
    ax5.set_ylabel('x₂', fontsize=11)
    ax5.set_title('Phase Portrait (with Directions)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8, loc='best')
    ax5.axis('equal')
    
    # 6. Eigenvalue plot
    ax6 = plt.subplot(2, 3, 6)
    for name, data in solutions.items():
        A = data['A']
        color = data['color']
        eigenvalues = np.linalg.eigvals(A)
        ax6.plot(eigenvalues.real, eigenvalues.imag, 'o', color=color, 
                markersize=12, label=name.split('(')[0].strip(),
                markeredgecolor='black', markeredgewidth=1)
    
    # Draw unit circle (for continuous time, stability boundary is imaginary axis)
    ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=1)
    ax6.axvline(x=0, color='k', linestyle='-', alpha=0.5, linewidth=2)
    
    # Shade stable region
    xlim = ax6.get_xlim()
    if xlim[0] < 0:
        ax6.axvspan(xlim[0], 0, alpha=0.1, color='green', label='Stable')
    if xlim[1] > 0:
        ax6.axvspan(0, xlim[1], alpha=0.1, color='red', label='Unstable')
    
    ax6.set_xlabel('Real part', fontsize=11)
    ax6.set_ylabel('Imaginary part', fontsize=11)
    ax6.set_title('Eigenvalues in Complex Plane', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8, loc='best')
    ax6.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Create separate figure for vector fields of each system
    n_systems = len(solutions)
    fig2 = plt.figure(figsize=(8 * n_systems, 7))
    fig2.suptitle('Vector Fields for Each System', fontsize=16, fontweight='bold')
    
    for idx, (name, data) in enumerate(solutions.items()):
        A = data['A']
        sol = data['sol']
        color = data['color']
        
        ax = plt.subplot(1, n_systems, idx + 1)
        
        # Plot vector field
        plot_vector_field(ax, A, x1_range, x2_range, n_points=20, scale=30)
        
        # Overlay trajectory
        ax.plot(sol.y[0, :], sol.y[1, :], color=color, linewidth=3, alpha=0.9, 
               label='Trajectory', zorder=5)
        
        # Mark start and end
        ax.plot(x0[0], x0[1], 's', color='black', markersize=14, 
               label='x₀', zorder=10, markeredgecolor='white', markeredgewidth=2)
        ax.plot(sol.y[0, -1], sol.y[1, -1], 'o', color=color, markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='Final', zorder=10)
        ax.plot(0, 0, 'kx', markersize=14, markeredgewidth=3, label='Origin', zorder=10)
        
        # Format plot
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, zorder=1)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, zorder=1)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(f'{name}\n(Vector Field: dx/dt = Ax)', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9, loc='best')
        ax.axis('equal')
        ax.set_xlim(x1_range)
        ax.set_ylim(x2_range)
    
    plt.tight_layout()
    plt.show()
    
    print("="*70)
    print("Simulation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
