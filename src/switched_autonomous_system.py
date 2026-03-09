"""
Switched autonomous system with two states and two subsystems.
Implements forced switching at x1 = 0 and s = cx1 + x2 = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.patches import Patch


class SwitchedSystem:
    """
    Switched autonomous system with forced switching conditions.
    """
    
    def __init__(self, A1, A2, c):
        """
        Initialize switched system.
        
        Parameters
        ----------
        A1 : np.ndarray
            System matrix for subsystem 1 (2x2)
        A2 : np.ndarray
            System matrix for subsystem 2 (2x2)
        c : float
            Parameter for switching surface s = cx1 + x2
        """
        self.A1 = A1
        self.A2 = A2
        self.c = c
        self.switching_times = []
        self.switching_states = []
        self.switching_surfaces = []  # Which surface triggered the switch
        
    def get_initial_subsystem(self, x0):
        """
        Determine initial subsystem based on initial state location.
        
        Rules:
        - Start in subsystem 1 if (s > 0 and x1 < 0) or (s < 0 and x1 > 0)
        - Start in subsystem 2 otherwise
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state [x1, x2]
            
        Returns
        -------
        int
            Initial subsystem (1 or 2)
        """
        x1, x2 = x0
        s = self.c * x1 + x2
        
        if (s > 0 and x1 < 0) or (s < 0 and x1 > 0):
            return 1
        else:
            return 2
    
    def dynamics(self, t, x, subsystem):
        """
        System dynamics for a given subsystem.
        
        Parameters
        ----------
        t : float
            Time
        x : np.ndarray
            State [x1, x2]
        subsystem : int
            Active subsystem (1 or 2)
            
        Returns
        -------
        np.ndarray
            State derivative
        """
        if subsystem == 1:
            return self.A1 @ x
        else:
            return self.A2 @ x
    
    def check_switching(self, x):
        """
        Check if a switching condition is met.
        
        Switching occurs at:
        - x1 = 0
        - s = cx1 + x2 = 0
        
        Parameters
        ----------
        x : np.ndarray
            Current state [x1, x2]
            
        Returns
        -------
        tuple
            (should_switch, surface_name) where surface_name is 'x1=0', 's=0', or None
        """
        x1, x2 = x
        s = self.c * x1 + x2
        
        # Check switching surfaces with a small tolerance
        tol = 1e-6
        
        if abs(x1) < tol:
            return True, 'x1=0'
        elif abs(s) < tol:
            return True, 's=0'
        else:
            return False, None
    

    
    def simulate(self, x0, t_span, max_switches=50, min_dwell_time=1e-3):
        """
        Simulate the switched system with forced switching.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state [x1, x2]
        t_span : tuple
            Time span (t_start, t_end)
        max_switches : int
            Maximum number of switches to prevent infinite loops
        min_dwell_time : float
            Minimum time between switches to prevent Zeno behavior
            
        Returns
        -------
        dict
            Dictionary containing:
            - 't': time array
            - 'x': state array (2, n_points)
            - 'subsystem': active subsystem at each time point
            - 'switching_times': times when switches occurred
            - 'switching_states': states at switching times
            - 'switching_surfaces': which surface caused each switch
        """
        # Determine initial subsystem
        current_subsystem = self.get_initial_subsystem(x0)
        
        # Storage for results
        t_all = []
        x_all = []
        subsystem_all = []
        
        t_current = t_span[0]
        x_current = x0
        n_switches = 0
        
        # Create event functions with proper attributes
        def event_x1(t, x):
            """Event function for x1 = 0 crossing."""
            return x[0]
        
        def event_s(t, x):
            """Event function for s = cx1 + x2 = 0 crossing."""
            return self.c * x[0] + x[1]
        
        # Set event function attributes
        event_x1.terminal = True
        event_x1.direction = 0  # Detect all crossings
        event_s.terminal = True
        event_s.direction = 0
        
        print(f"Initial state: x0 = [{x0[0]:.4f}, {x0[1]:.4f}]")
        print(f"Initial s = cx1 + x2 = {self.c * x0[0] + x0[1]:.4f}")
        print(f"Starting in subsystem {current_subsystem}")
        print()
        
        while t_current < t_span[1] and n_switches < max_switches:
            # Enforce minimum dwell time - simulate at least this far
            t_next_earliest = t_current + min_dwell_time
            
            # Simulate until next switching event
            sol = solve_ivp(
                lambda t, x: self.dynamics(t, x, current_subsystem),
                (t_current, t_span[1]),
                x_current,
                method='RK45',
                events=[event_x1, event_s],
                dense_output=True,
                rtol=1e-8,
                atol=1e-10
            )
            
            # Store results
            t_all.append(sol.t)
            x_all.append(sol.y)
            subsystem_all.append(np.full(len(sol.t), current_subsystem))
            
            # Check if an event occurred
            event_occurred = False
            if sol.t_events[0].size > 0 or sol.t_events[1].size > 0:
                # Determine which event occurred first
                event_times = []
                event_indices = []
                if sol.t_events[0].size > 0:
                    event_times.append(sol.t_events[0][0])
                    event_indices.append(0)
                if sol.t_events[1].size > 0:
                    event_times.append(sol.t_events[1][0])
                    event_indices.append(1)
                
                # Take the first event
                first_event_idx = event_indices[np.argmin(event_times)]
                t_switch = min(event_times)
                x_switch = sol.sol(t_switch)
                
                if first_event_idx == 0:
                    surface_name = 'x1=0'
                else:
                    surface_name = 's=0'
                
                self.switching_times.append(t_switch)
                self.switching_states.append(x_switch)
                self.switching_surfaces.append(surface_name)
                
                print(f"Switch {n_switches + 1} at t = {t_switch:.4f} s")
                print(f"  State: x = [{x_switch[0]:.4f}, {x_switch[1]:.4f}]")
                print(f"  Surface: {surface_name}")
                print(f"  Subsystem {current_subsystem} -> {3 - current_subsystem}")
                
                # Switch subsystem
                current_subsystem = 3 - current_subsystem  # Toggle between 1 and 2
                t_current = t_switch
                x_current = x_switch
                n_switches += 1
            else:
                # No more events, simulation complete
                break
        
        if n_switches >= max_switches:
            print(f"\nWarning: Maximum number of switches ({max_switches}) reached.")
        
        # Concatenate all segments
        t_final = np.concatenate(t_all)
        x_final = np.concatenate(x_all, axis=1)
        subsystem_final = np.concatenate(subsystem_all)
        
        return {
            't': t_final,
            'x': x_final,
            'subsystem': subsystem_final,
            'switching_times': np.array(self.switching_times),
            'switching_states': np.array(self.switching_states).T if self.switching_states else np.array([[], []]),
            'switching_surfaces': self.switching_surfaces
        }


def plot_vector_field(ax, A, x_range, y_range, n_points=15, color='gray', alpha=0.4, scale=None):
    """
    Plot the vector field for the system dx/dt = Ax
    """
    x1 = np.linspace(x_range[0], x_range[1], n_points)
    x2 = np.linspace(y_range[0], y_range[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Compute vector field
    DX1 = A[0, 0] * X1 + A[0, 1] * X2
    DX2 = A[1, 0] * X1 + A[1, 1] * X2
    
    # Normalize vectors for better visualization
    magnitude = np.sqrt(DX1**2 + DX2**2)
    magnitude = np.where(magnitude == 0, 1, magnitude)
    
    # Plot vector field
    ax.quiver(X1, X2, DX1/magnitude, DX2/magnitude, 
             magnitude, cmap='viridis', alpha=alpha, scale=scale, width=0.003,
             headwidth=4, headlength=5, headaxislength=4.5)


def plot_switching_surfaces(ax, c, x_range, y_range):
    """
    Plot the switching surfaces x1 = 0 and s = cx1 + x2 = 0.
    """
    # x1 = 0 (vertical line)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='x₁ = 0')
    
    # s = cx1 + x2 = 0  =>  x2 = -c*x1
    x1_line = np.array(x_range)
    x2_line = -c * x1_line
    ax.plot(x1_line, x2_line, 'b--', linewidth=2, alpha=0.7, label=f's = {c:.1f}x₁ + x₂ = 0')


def main():
    # System parameters
    # Define two subsystem matrices
    a = 1
    b = 2
    lam = 4
    A1 = np.array([
        [0.0, 1.0],
        [-a+lam, b]
    ])
    
    A2 = np.array([
        [0.0, 1.0],
        [-a-lam, b]
    ])
    
    # Switching surface parameter
    c = 0.1
    
    # Initial condition
    x0 = np.array([1.0, 1.0])
    
    # Time settings
    t_start = 0.0
    t_end = 1.0
    
    print("="*70)
    print("SWITCHED AUTONOMOUS SYSTEM SIMULATION")
    print("="*70)
    print(f"Switching parameter: c = {c}")
    print(f"Time span: [{t_start:.1f}, {t_end:.1f}] s")
    print()
    print("Subsystem 1 matrix A1:")
    print(f"  {A1[0]}")
    print(f"  {A1[1]}")
    print(f"Eigenvalues: {np.linalg.eigvals(A1)}")
    print()
    print("Subsystem 2 matrix A2:")
    print(f"  {A2[0]}")
    print(f"  {A2[1]}")
    print(f"Eigenvalues: {np.linalg.eigvals(A2)}")
    print()
    print("Switching surfaces:")
    print("  - x₁ = 0")
    print(f"  - s = {c}x₁ + x₂ = 0")
    print()
    
    # Create and simulate switched system
    system = SwitchedSystem(A1, A2, c)
    results = system.simulate(x0, (t_start, t_end))
    
    print()
    print(f"Total switches: {len(results['switching_times'])}")
    print("="*70)
    
    # Extract results
    t = results['t']
    x = results['x']
    subsystem = results['subsystem']
    t_switch = results['switching_times']
    x_switch = results['switching_states']
    
    # Determine plot ranges
    margin = 0.3
    x1_range = (np.min(x[0, :]) - margin, np.max(x[0, :]) + margin)
    x2_range = (np.min(x[1, :]) - margin, np.max(x[1, :]) + margin)
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Switched Autonomous System with Forced Switching', fontsize=16, fontweight='bold')
    
    # 1. Phase portrait with trajectory
    ax1 = plt.subplot(2, 3, 1)
    
    # Color trajectory by active subsystem
    for i in range(len(t) - 1):
        if subsystem[i] == 1:
            color = 'blue'
        else:
            color = 'red'
        ax1.plot(x[0, i:i+2], x[1, i:i+2], color=color, linewidth=2, alpha=0.7)
    
    # Plot switching surfaces
    plot_switching_surfaces(ax1, c, x1_range, x2_range)
    
    # Mark switching points
    if len(x_switch) > 0 and x_switch.size > 0:
        ax1.plot(x_switch[0, :], x_switch[1, :], 'go', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5, label='Switching points', zorder=10)
    
    # Mark initial and final points
    ax1.plot(x0[0], x0[1], 's', color='black', markersize=12, 
            label='x₀', zorder=11, markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(x[0, -1], x[1, -1], 'p', color='purple', markersize=12, 
            label='Final', zorder=11, markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, label='Origin', zorder=10)
    
    # Add legend entries for subsystems
    ax1.plot([], [], 'b-', linewidth=2, label='Subsystem 1')
    ax1.plot([], [], 'r-', linewidth=2, label='Subsystem 2')
    
    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('x₂', fontsize=11)
    ax1.set_title('Phase Portrait with Switching Surfaces', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    ax1.axis('equal')
    ax1.set_xlim(x1_range)
    ax1.set_ylim(x2_range)
    
    # 2. State x₁ vs time
    ax2 = plt.subplot(2, 3, 2)
    for i in range(len(t) - 1):
        color = 'blue' if subsystem[i] == 1 else 'red'
        ax2.plot(t[i:i+2], x[0, i:i+2], color=color, linewidth=2)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='x₁ = 0')
    
    # Mark switching times
    if len(t_switch) > 0:
        for ts in t_switch:
            ax2.axvline(x=ts, color='green', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('x₁', fontsize=11)
    ax2.set_title('State x₁ vs Time', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')
    
    # 3. State x₂ vs time
    ax3 = plt.subplot(2, 3, 3)
    for i in range(len(t) - 1):
        color = 'blue' if subsystem[i] == 1 else 'red'
        ax3.plot(t[i:i+2], x[1, i:i+2], color=color, linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Mark switching times
    if len(t_switch) > 0:
        for ts in t_switch:
            ax3.axvline(x=ts, color='green', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('x₂', fontsize=11)
    ax3.set_title('State x₂ vs Time', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Switching surface s = cx1 + x2 vs time
    ax4 = plt.subplot(2, 3, 4)
    s_values = c * x[0, :] + x[1, :]
    for i in range(len(t) - 1):
        color = 'blue' if subsystem[i] == 1 else 'red'
        ax4.plot(t[i:i+2], s_values[i:i+2], color=color, linewidth=2)
    ax4.axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='s = 0')
    
    # Mark switching times
    if len(t_switch) > 0:
        for ts in t_switch:
            ax4.axvline(x=ts, color='green', linestyle=':', alpha=0.5)
    
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel(f's = {c}x₁ + x₂', fontsize=11)
    ax4.set_title('Switching Surface Value vs Time', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8, loc='best')
    
    # 5. Active subsystem vs time
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(t, subsystem, 'k-', linewidth=2, drawstyle='steps-post')
    ax5.fill_between(t, 0, subsystem, where=(subsystem == 1), 
                     color='blue', alpha=0.3, step='post', label='Subsystem 1')
    ax5.fill_between(t, 0, subsystem, where=(subsystem == 2), 
                     color='red', alpha=0.3, step='post', label='Subsystem 2')
    
    # Mark switching times
    if len(t_switch) > 0:
        for ts in t_switch:
            ax5.axvline(x=ts, color='green', linestyle=':', alpha=0.7)
    
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Active Subsystem', fontsize=11)
    ax5.set_title('Subsystem Switching', fontweight='bold')
    ax5.set_ylim([0.5, 2.5])
    ax5.set_yticks([1, 2])
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.legend(fontsize=8, loc='best')
    
    # 6. Distance from origin vs time
    ax6 = plt.subplot(2, 3, 6)
    distances = np.linalg.norm(x, axis=0)
    for i in range(len(t) - 1):
        color = 'blue' if subsystem[i] == 1 else 'red'
        ax6.plot(t[i:i+2], distances[i:i+2], color=color, linewidth=2)
    
    # Mark switching times
    if len(t_switch) > 0:
        for ts in t_switch:
            ax6.axvline(x=ts, color='green', linestyle=':', alpha=0.5)
    
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('||x||₂', fontsize=11)
    ax6.set_title('Distance from Origin', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Create figure with vector fields
    fig2 = plt.figure(figsize=(18, 7))
    fig2.suptitle('Vector Fields for Both Subsystems', fontsize=16, fontweight='bold')
    
    # Subsystem 1 vector field
    ax_v1 = plt.subplot(1, 3, 1)
    plot_vector_field(ax_v1, A1, x1_range, x2_range, n_points=20, scale=30)
    
    # Overlay trajectory segments where subsystem 1 is active
    for i in range(len(t) - 1):
        if subsystem[i] == 1:
            ax_v1.plot(x[0, i:i+2], x[1, i:i+2], 'b-', linewidth=2.5, alpha=0.8)
    
    plot_switching_surfaces(ax_v1, c, x1_range, x2_range)
    ax_v1.plot(x0[0], x0[1], 's', color='black', markersize=12, label='x₀', zorder=10)
    ax_v1.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, label='Origin', zorder=10)
    
    ax_v1.set_xlabel('x₁', fontsize=11)
    ax_v1.set_ylabel('x₂', fontsize=11)
    ax_v1.set_title('Subsystem 1 Vector Field\n(dx/dt = A₁x)', fontweight='bold')
    ax_v1.grid(True, alpha=0.2)
    ax_v1.legend(fontsize=8, loc='best')
    ax_v1.axis('equal')
    ax_v1.set_xlim(x1_range)
    ax_v1.set_ylim(x2_range)
    
    # Subsystem 2 vector field
    ax_v2 = plt.subplot(1, 3, 2)
    plot_vector_field(ax_v2, A2, x1_range, x2_range, n_points=20, scale=30)
    
    # Overlay trajectory segments where subsystem 2 is active
    for i in range(len(t) - 1):
        if subsystem[i] == 2:
            ax_v2.plot(x[0, i:i+2], x[1, i:i+2], 'r-', linewidth=2.5, alpha=0.8)
    
    plot_switching_surfaces(ax_v2, c, x1_range, x2_range)
    ax_v2.plot(x0[0], x0[1], 's', color='black', markersize=12, label='x₀', zorder=10)
    ax_v2.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, label='Origin', zorder=10)
    
    ax_v2.set_xlabel('x₁', fontsize=11)
    ax_v2.set_ylabel('x₂', fontsize=11)
    ax_v2.set_title('Subsystem 2 Vector Field\n(dx/dt = A₂x)', fontweight='bold')
    ax_v2.grid(True, alpha=0.2)
    ax_v2.legend(fontsize=8, loc='best')
    ax_v2.axis('equal')
    ax_v2.set_xlim(x1_range)
    ax_v2.set_ylim(x2_range)
    
    # Combined view
    ax_v3 = plt.subplot(1, 3, 3)
    
    # Plot both vector fields with different colors
    x1_grid = np.linspace(x1_range[0], x1_range[1], 20)
    x2_grid = np.linspace(x2_range[0], x2_range[1], 20)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Subsystem 1 in blue
    DX1_1 = A1[0, 0] * X1 + A1[0, 1] * X2
    DX2_1 = A1[1, 0] * X1 + A1[1, 1] * X2
    mag1 = np.sqrt(DX1_1**2 + DX2_1**2)
    mag1 = np.where(mag1 == 0, 1, mag1)
    ax_v3.quiver(X1, X2, DX1_1/mag1, DX2_1/mag1, 
                color='blue', alpha=0.3, scale=30, width=0.003,
                headwidth=4, headlength=5, headaxislength=4.5)
    
    # Subsystem 2 in red
    DX1_2 = A2[0, 0] * X1 + A2[0, 1] * X2
    DX2_2 = A2[1, 0] * X1 + A2[1, 1] * X2
    mag2 = np.sqrt(DX1_2**2 + DX2_2**2)
    mag2 = np.where(mag2 == 0, 1, mag2)
    ax_v3.quiver(X1, X2, DX1_2/mag2, DX2_2/mag2, 
                color='red', alpha=0.3, scale=30, width=0.003,
                headwidth=4, headlength=5, headaxislength=4.5)
    
    # Overlay complete trajectory
    for i in range(len(t) - 1):
        color = 'blue' if subsystem[i] == 1 else 'red'
        ax_v3.plot(x[0, i:i+2], x[1, i:i+2], color=color, linewidth=2.5, alpha=0.9)
    
    if len(x_switch) > 0 and x_switch.size > 0:
        ax_v3.plot(x_switch[0, :], x_switch[1, :], 'go', markersize=10, 
                  markeredgecolor='black', markeredgewidth=1.5, label='Switches', zorder=10)
    
    plot_switching_surfaces(ax_v3, c, x1_range, x2_range)
    ax_v3.plot(x0[0], x0[1], 's', color='black', markersize=12, label='x₀', zorder=11)
    ax_v3.plot(x[0, -1], x[1, -1], 'p', color='purple', markersize=12, 
              label='Final', zorder=11, markeredgecolor='black', markeredgewidth=1.5)
    ax_v3.plot(0, 0, 'kx', markersize=12, markeredgewidth=3, label='Origin', zorder=10)
    
    ax_v3.set_xlabel('x₁', fontsize=11)
    ax_v3.set_ylabel('x₂', fontsize=11)
    ax_v3.set_title('Combined: Switched Trajectory', fontweight='bold')
    ax_v3.grid(True, alpha=0.2)
    ax_v3.legend(fontsize=8, loc='best')
    ax_v3.axis('equal')
    ax_v3.set_xlim(x1_range)
    ax_v3.set_ylim(x2_range)
    
    plt.tight_layout()
    plt.show()
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
