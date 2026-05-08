# MatterSim GUI: Materials Simulation & Property Prediction

A Graphical User Interface (GUI) application for calculating and simulating material properties, phase diagrams, and phonon dispersion.

## Features
- **ASE (Atomic Simulation Environment) Integration:** Robust support for building and manipulating atomic structures.
- **Phase Diagram & Equation of State (EOS):** Supports Birch-Murnaghan equation of state fitting to determine equilibrium volume and bulk modulus.
- **Phonon Dispersion Processing:** Calculates phonon frequencies and thermodynamic properties using Phonopy.
- **Vacuum Box Mechanism:** Accurately handles vacuum boundaries in nanoparticle and molecular simulations.

## Screenshots
*(Insert GUI screenshots or result plots here)*
<!-- Example image: ![MatterSim GUI](docs/gui_screenshot.png) -->

## Installation

Requirements: Python 3.8+ and related dependencies.

1. Clone this repository to your local machine.
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```
2. Run the application via Python:
   ```bash
   python "Lattice constant_Prediction.py"
   ```
3. The GUI will appear. Select a material, choose a calculation mode, and click "Run" to view the results.
