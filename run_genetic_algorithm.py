"""
Genetic Algorithm with AFPO (Age-Fitness Pareto Optimization) for robot morphology evolution.

GA components:
- Selection: AFPO - parents chosen from Pareto front on (fitness, age)
- Crossover: uniform crossover of two parent masks
- Mutation: flip one random voxel
- Replacement: Pareto-based - keep non-dominated individuals, fill with offspring

AFPO prevents bloat by letting young (low-fitness) individuals compete with old (high-fitness) ones.
"""

from simulator import Simulator
from utils import load_config
from argparse import ArgumentParser
from robot import robot_from_mask, sample_mask, mutate_mask, crossover_mask
import numpy as np
import time
from tqdm import tqdm


def pareto_rank(fitness, age):
    """
    Pareto ranking on (fitness, age). Maximize fitness, minimize age.
    Returns rank for each individual (0 = Pareto front, 1 = next front, etc.)
    A dominates B if fitness_A >= fitness_B AND age_A <= age_B (strict in at least one).
    """
    n = len(fitness)
    rank = np.full(n, -1)
    current_rank = 0
    remaining = np.ones(n, dtype=bool)

    while np.any(remaining):
        idx = np.where(remaining)[0]
        front = []
        for i in idx:
            dominated = False
            for j in idx:
                if i == j:
                    continue
                # j dominates i if: fitness_j >= fitness_i AND age_j <= age_i, strict in one
                if fitness[j] >= fitness[i] and age[j] <= age[i]:
                    if fitness[j] > fitness[i] or age[j] < age[i]:
                        dominated = True
                        break
            if not dominated:
                front.append(i)
        for i in front:
            rank[i] = current_rank
            remaining[i] = False
        current_rank += 1

    return rank


def select_parents(population, fitness, age, n_select):
    """Select n_select individuals for parenting using AFPO (Pareto-based selection)."""
    rank = pareto_rank(fitness, age)
    # Prefer lower rank (Pareto front). Within same rank, prefer higher fitness.
    # Tournament: pick 2, choose one with better (lower rank, or higher fitness if same rank)
    parents = []
    n = len(population)
    for _ in range(n_select):
        i, j = np.random.randint(n), np.random.randint(n)
        if rank[i] < rank[j]:
            parents.append(i)
        elif rank[i] > rank[j]:
            parents.append(j)
        else:
            parents.append(i if fitness[i] >= fitness[j] else j)
    return parents


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--generations", type=int, default=10, help="Number of GA generations")
    parser.add_argument("--mutation-rate", type=float, default=1.0, help="Probability of mutating each offspring")
    args = parser.parse_args()

    config = load_config(args.config)
    n_generations = args.generations
    mutation_rate = args.mutation_rate
    pop_size = config["simulator"]["n_sims"]

    np.random.seed(config["seed"])
    start_time = time.perf_counter()

    # Population: list of (robot, age)
    population = [robot_from_mask(sample_mask(0.55)) for _ in range(pop_size)]
    age = np.zeros(pop_size, dtype=int)
    fitness = np.zeros(pop_size)

    # Evaluate initial population
    num_masses = [r["n_masses"] for r in population]
    num_springs = [r["n_springs"] for r in population]
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
    sim_init.initialize([r["masses"] for r in population], [r["springs"] for r in population])
    fitness = sim_init.train()[:, -1].copy()
    fitness = np.nan_to_num(fitness, nan=-np.inf, posinf=0.0, neginf=-np.inf)
    fitness_history = [fitness.copy()]

    print(f"Genetic Algorithm + AFPO: pop={pop_size}, {n_generations} generations")
    print(f"Initial best fitness: {fitness.max():.2f}")
    print("-" * 50)

    for gen in tqdm(range(n_generations - 1), desc="Generation"):
        # Age current population
        age += 1

        # Select parents (AFPO)
        parents = select_parents(population, fitness, age, pop_size * 2)

        # Create offspring: crossover + mutation
        offspring = []
        for k in range(pop_size):
            i, j = parents[2 * k], parents[2 * k + 1]
            child_mask = crossover_mask(population[i]["mask"], population[j]["mask"])
            if np.random.random() < mutation_rate:
                child_mask = mutate_mask(child_mask)
            offspring.append(robot_from_mask(child_mask))

        # Evaluate offspring (age 0)
        num_masses = [r["n_masses"] for r in offspring]
        num_springs = [r["n_springs"] for r in offspring]
        max_num_masses = max(num_masses)
        max_num_springs = max(num_springs)
        config["simulator"]["n_masses"] = max_num_masses
        config["simulator"]["n_springs"] = max_num_springs

        sim_off = Simulator(
            sim_config=config["simulator"],
            taichi_config=config["taichi"],
            seed=config["seed"] + gen + 1,
            needs_grad=True,
        )
        sim_off.initialize([r["masses"] for r in offspring], [r["springs"] for r in offspring])
        offspring_fitness = sim_off.train()[:, -1]
        offspring_fitness = np.nan_to_num(offspring_fitness, nan=-np.inf, posinf=0.0, neginf=-np.inf)
        offspring_age = np.zeros(pop_size, dtype=int)

        # AFPO replacement: combine population + offspring, Pareto select N
        combined_pop = population + offspring
        combined_fitness = np.concatenate([fitness, offspring_fitness])
        combined_age = np.concatenate([age, offspring_age])

        rank = pareto_rank(combined_fitness, combined_age)
        # Sort by rank, then by fitness within rank
        order = np.lexsort((-combined_fitness, rank))
        survivors = order[:pop_size]

        population = [combined_pop[i] for i in survivors]
        fitness = combined_fitness[survivors]
        age = combined_age[survivors]

        fitness_history.append(fitness.copy())

    fitness_history = np.array(fitness_history).T  # (pop_size, n_generations)

    # Save results
    np.save("fitness_history.npy", fitness_history)

    # Rank by fitness, save top 3
    ranking = np.argsort(fitness)[::-1]
    top_3_idxs = ranking[:3]
    top_3_robots = [population[i] for i in top_3_idxs]

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
        np.save(f"robot_ga_{i}.npy", robot)

    elapsed = time.perf_counter() - start_time
    best_fitness = fitness[ranking[0]]
    print(f"\nDone! Best fitness: {best_fitness:.2f}")
    print(f"Top 3 robots saved to robot_ga_0.npy, robot_ga_1.npy, robot_ga_2.npy")
    print(f"\nBack-of-the-napkin: {n_generations} gens took {elapsed/60:.1f} min")
    print(f"  → ~{elapsed/n_generations:.0f}s per gen → 20 gens ≈ {20*elapsed/n_generations/60:.1f} min")
    print("Run plot_fitness.ipynb to visualize, or python visualizer.py --input robot_ga_0.npy --port 5001")
