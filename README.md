# Interactive Evolutionary Multiobjective Optimization of Primer Design with Uncertain Objectives

This repository contains the implementation for the paper [Hybrid Surrogate Assisted Evolutionary Multiobjective Reinforcement Learning for Continuous Robot Control](https://link.springer.com/chapter/10.1007/978-3-031-56855-8_4) (**EVOSTAR 2024**).
In this paper, we propose a hybrid multiobjective policy optimization approach for solving multiobjective reinforcement learning (MORL) problems with continuous actions. Our approach combines the faster convergence of multiobjective policy gradient (MOPG) and a surrogate-assisted multiobjective evolutionary algorithm (MOEA) to produce a dense set of Pareto optimal policies. 
The project utilizes the work on [Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control](http://people.csail.mit.edu/jiex/papers/PGMORL/) (**ICML 2020**) or PGMORL as the base MORL algorithm. Refer to PGMORL [Github](https://github.com/mit-gfx/PGMORL) implementation for more details.
We tested our algorithm with the MUJOCO benchmarks proposed in the PGMORL paper and our multiobjective implementation of a simple autonomous driving environment [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv). 

## Installation

### Requirements:
* **DESDEO**: The project relies on the desdeo-emo, desdeo_tools, and desdeo-problems packages for surrogate assisted multiobjective optimization used and defining the optimization problem.
* **Python**: Version 3.7 or up
* **PyTorch Version**: >= 1.3.0.
* **MuJoCo**: install mujoco and mujoco-py of version 2.0 by following the instructions in [mujoco-py](<https://github.com/openai/mujoco-py>).

### Clone Repository
`gh repo clone amrzr/SA-MOEAMOPG`

### Installation Process:
* Create conda environment by
`conda env create -f env.yml`

* Activate the conda environment
` conda activate samoea-morl`

