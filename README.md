# SpinEq
**A program to compute chemical equilibria along with NMR magnetization evolution.**

## What it does
SpinEq allows to simulate in the same times chemical kinetics and NMR experiments conducted in presence of a chemical reaction.

The system is describes exclusively in terms of kinetics flux, with species undergoing different kinetics steps, and with NMR spins attached to the molecules. 

The kinetic systems is first described, then the initial state is given (concentration of each species and magnetization of each spins), then the system is inegrated over time, with the possibility to store the trajectories of all quantities.

There is no hypothesis of stationarity or equilibrium, and off-equilibrium systems can be fully studied.
However, one limitation for the moment is that transverse magnetisation can be computed only on stationary systems.

it based on 3 kind of top-level objects:

- `Eq()` is for chemical reactions  simulate equilibria and kinetics, but not NMR
- `SpinSys()` simulates longitudinal magnetisation of a spin system
- `NMR()` a combination of both objects above.

with a set of tools allowing to set and manipulate them, 

as well as on one low-level object:

- `Flux()` the elementary object holding a chemical kinetics -

There are also utility objects:

- `Rates()` which allow to compute $\rho$ and $\sigma$ relaxation parameters from tumbling rates.
- `gen_fid()` which generates a fid given a set of frequencies, T~2~ and the kinetics interconversion rates.

When setting-up, note that

- all concentration are in M
- all times are in sec

## Examples
look at `examples.py` for examples, with the graphical results in `examples.ipynb`

It contains:

- protein - ligand interactions
- comparison of selective and non-selective T~1~ experiments
- nOe in a 2 spins and in a n-spins system
- spectra of species slow and fast exchange
- Saturation Transfer Difference (STD) in a simple system, and with 2 sites in competitions

## Licence
This code is release under the [CeCILL 2.1 licence](Licence_CeCILL_V2.1-en.txt) - equivalent to GPL

