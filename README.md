## Overview
Here is the video: https://drive.google.com/file/d/1o4XIP0TX3874m3taJlQsFj67ZRn5Z7N7/view?usp=sharing 


This repository provides a physical simulation platform for studying automatic design of robots or [virtual creatures](https://www.nature.com/articles/s42256-019-0102-8). It is based largely on the paper, [Evolution and learning in differentiable robots](https://sites.google.com/view/eldir). By abstracting away the physical simulation and control optimization details, this codebase makes it possible to quickly iterate on algorithms for morphological design.

## Installation

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) if you do not already have it. 
2. Create a new environment: `conda create --name alife-sim` (you can replace "alife-sim" with another name as you like).
3. Activate the environment: `conda activate alife-sim`.
4. Install Python: `conda install python=3.12`
5. Install Taichi: `pip install taichi==1.7.3`
6. Install other packages: `pip install tqdm scipy pyaml flask ipykernel matplotlib`

## Usage

1. Review the code in `run.py`. It shows an example of how to interface with the simulator.
2. Next review `config.yaml`. This includes a number of parameters, only a small number of which you should consider modifying. 
3. Review `robot.py`. This code illustrates how random robot designs can be sampled and explains the key constraints to keep in mind when representing robots for the simulator. You can also visualize designs in `visualize_robots.ipynb`. 
4. Finally, try to run the code: `python run.py`. This will generate some results files that you can visualize with `plot_fitness.ipynb` and `visualizer.py`.

## Parallel Hill Climber (Evolutionary Morphology Search)

To run the parallel hill climber for evolving robot morphologies:

```bash
python run_hill_climber.py --generations 10
```

This runs 16 independent hill climbers (one per `n_sims` in config), each mutating and improving their robot morphology over generations. Results are saved in the same format as `run.py`—use `plot_fitness.ipynb` to view fitness over generations and `visualizer.py` to watch the best robots.

## Genetic Algorithm + AFPO

To run the genetic algorithm with Age-Fitness Pareto Optimization:

```bash
python run_genetic_algorithm.py --generations 10
```

This uses crossover (two parents → child), mutation, and AFPO for selection—keeping a Pareto front on (fitness, age) so young solutions can compete with old ones and prevent bloat. Same output format as above. 
