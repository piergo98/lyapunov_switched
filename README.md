# Switched Autonomous System (Lyapunov / Forced Switching)

A small Python project that implements and visualizes a switched autonomous system with two 2D linear subsystems and forced switching on the surfaces x1 = 0 and s = c*x1 + x2 = 0.

Features
- Simulate switched dynamics with event-based switching using `scipy.integrate.solve_ivp`.
- Plot phase portraits, switching surfaces, subsystem vector fields, and time histories.

Requirements
- Python 3.8+ (Linux / macOS / Windows)
- numpy
- scipy
- matplotlib

Install

It is recommended to use a virtual environment. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy matplotlib
```

Run

From the project root run the main example script:

```bash
python src/switched_autonomous_system.py
```

This will run the simulation with a default example initial condition and show several figures (phase portrait, time series, vector fields).

Files of interest
- `src/switched_autonomous_system.py`: main example and plotting utilities.
- `src/switched_clf.py`, `src/switched_clf_ss.py`, `src/clf_ss.py`, `src/autonomous_systems_comparison.py`: additional modules and experiments (see code comments).

Notes
- The simulation uses event detection to find exact switching times. If you modify tolerances or integrator settings you may affect switching detection.
- The code is intended for experimentation and visualization; adapt or refactor into packages/modules if you need to integrate it into larger projects.

License
- No license provided. Add a `LICENSE` file if you want to specify reuse terms.

If you want, I can:
- Add a `requirements.txt` or `pyproject.toml`.
- Add example command-line arguments to select initial conditions or plotting options.
