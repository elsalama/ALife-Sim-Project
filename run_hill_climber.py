"""
Parallel Hill Climber for robot morphology evolution.

Each of N independent hill climbers maintains a robot morphology. Each generation:
1. Mutate each climber's morphology (flip one voxel in the mask)
2. Train control for all mutants in parallel
3. If mutant is better than current, keep it; else keep current
4. Repeat for n_generations
"""

from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import robot_from_mask, sample_mask, mutate_mask
import numpy as np
import time
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--generations", type=int, default=10, help="Number of hill climbing generations")
    args = parser.parse_args()

    config = load_config(args.config)
    n_generations = args.generations
    n_climbers = config["simulator"]["n_sims"]

    np.random.seed(config["seed"])
    start_time = time.perf_counter()

    # Initialize: each climber gets a random robot (with mask for mutation)
    climbers = [robot_from_mask(sample_mask(0.55)) for _ in range(n_climbers)]

    # Evaluate initial population
    num_masses = [r["n_masses"] for r in climbers]
    num_springs = [r["n_springs"] for r in climbers]
    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)
    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs

    sim_init = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True,
    )
    sim_init.initialize([r["masses"] for r in climbers], [r["springs"] for r in climbers])
    init_fitness = sim_init.train()[:, -1]
    climber_fitness = np.nan_to_num(init_fitness, nan=-np.inf, posinf=0.0, neginf=-np.inf).copy()
    fitness_history = [climber_fitness.copy()]

    print(f"Parallel Hill Climber: {n_climbers} climbers, {n_generations} generations")
    print(f"Initial best fitness: {climber_fitness.max():.2f}")
    print("-" * 50)

    for gen in tqdm(range(n_generations - 1), desc="Generation"):
        # Mutate each climber
        mutants = [robot_from_mask(mutate_mask(c["mask"])) for c in climbers]

        # Get max dimensions for this batch
        num_masses = [r["n_masses"] for r in mutants]
        num_springs = [r["n_springs"] for r in mutants]
        max_num_masses = max(num_masses)
        max_num_springs = max(num_springs)

        config["simulator"]["n_masses"] = max_num_masses
        config["simulator"]["n_springs"] = max_num_springs

        # Create simulator and train mutants
        simulator = Simulator(
            sim_config=config["simulator"],
            taichi_config=config["taichi"],
            seed=config["seed"] + gen,  # vary seed per gen for weight init
            needs_grad=True,
        )
        masses = [r["masses"] for r in mutants]
        springs = [r["springs"] for r in mutants]
        simulator.initialize(masses, springs)

        fitness_history_gen = simulator.train()
        mutant_fitness = np.nan_to_num(fitness_history_gen[:, -1], nan=-np.inf, posinf=0.0, neginf=-np.inf)

        # Hill climbing: keep mutant if better
        for i in range(n_climbers):
            if mutant_fitness[i] > climber_fitness[i]:
                climbers[i] = mutants[i]
                climber_fitness[i] = mutant_fitness[i]
            # else: keep current climber (climber_fitness unchanged)

        fitness_history.append(climber_fitness.copy())

    fitness_history = np.array(fitness_history).T  # (n_climbers, n_generations)

    # Save results (same format as run.py for compatibility with visualizer/plot_fitness)
    np.save("fitness_history.npy", fitness_history)

    # Rank by final fitness, save top 3
    ranking = np.argsort(climber_fitness)[::-1]
    top_3_idxs = ranking[:3]
    top_3_robots = [climbers[i] for i in top_3_idxs]

    # Re-run simulator for top 3 to get their trained control params
    num_masses = [r["n_masses"] for r in top_3_robots]
    num_springs = [r["n_springs"] for r in top_3_robots]
    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)
    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs
    config["simulator"]["n_sims"] = 3

    sim_final = Simulator(
        sim_config=config["simulator"],
        taichi_config=config["taichi"],
        seed=config["seed"],
        needs_grad=True,
    )
    sim_final.initialize(
        [r["masses"] for r in top_3_robots],
        [r["springs"] for r in top_3_robots],
    )
    sim_final.train()
    top_3_control_params = sim_final.get_control_params([0, 1, 2])

    for i in range(3):
        robot = top_3_robots[i]
        robot["control_params"] = top_3_control_params[i]
        robot["max_n_masses"] = max_num_masses
        robot["max_n_springs"] = max_num_springs
        np.save(f"robot_hc_{i}.npy", robot)

    elapsed = time.perf_counter() - start_time
    best_fitness = climber_fitness[ranking[0]]
    print(f"\nDone! Best fitness: {best_fitness:.2f}")
    print(f"Top 3 robots saved to robot_hc_0.npy, robot_hc_1.npy, robot_hc_2.npy")
    print(f"\nBack-of-the-napkin: {n_generations} gens took {elapsed/60:.1f} min")
    print(f"  → ~{elapsed/n_generations:.0f}s per gen → 20 gens ≈ {20*elapsed/n_generations/60:.1f} min")
    print("Run plot_fitness.ipynb to visualize, or python visualizer.py --input robot_hc_0.npy --port 5001")
