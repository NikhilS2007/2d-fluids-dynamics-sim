# 2D Fluids Dynamics Simulation

A simple CPU-based 2D "stable fluids" (Navier–Stokes) demonstration implemented in Python.
This repository contains a compact implementation (Jos Stam style) that simulates and
visualizes density (like smoke) using `numpy` and `matplotlib`.

## Files
- `fluid_sim.py`: Main simulation file and demo runner.

## Requirements
- Python 3.8 or newer
- `numpy`
- `matplotlib`

Install dependencies quickly with pip:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

Optionally create a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
```

## Run
Run the demo from the repository root:

```powershell
python fluid_sim.py
```

The program opens an interactive matplotlib window showing density. The demo runs
for a fixed number of frames and adds density/velocity in the grid center. Close the
plot window to end the demo early.

## Tweak parameters
Open `fluid_sim.py` and edit `run_demo()` to change:
- `N` — grid resolution (N x N). Larger values increase accuracy and CPU cost.
- `diff` — diffusion coefficient.
- `visc` — viscosity.
- `dt` — timestep.

## Notes
- This is a small educational demo optimized for clarity, not performance.
- No packaging or license is included. Use and modify freely.
