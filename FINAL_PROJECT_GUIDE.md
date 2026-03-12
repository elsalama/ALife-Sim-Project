# Final Project Video — Step-by-Step Guide

Follow these steps in order for your video. Record your screen as you go.

---

## Step 1: BEFORE EVOLUTION — Show a random robot

### 1a. Morphology (shape) in the notebook

1. Open `visualize_robots.ipynb`
2. Run **Cell 1** (defines `plot_robot`)
3. Run **Cell 2** (imports `sample_robot`)
4. Run **Cell 3** (`robot = sample_robot()`)

**What to say:** "This is a random robot before any evolution. Its shape is random."

### 1b. Random robot moving (poorly) in the visualizer

1. In terminal, run:
   ```bash
   python save_random_robot.py
   ```
2. Then run:
   ```bash
   python visualizer.py --input random_robot.npy --port 5001
   ```
3. Open http://localhost:5001 in your browser

**What to say:** "Here it is in the simulator. It has no trained controller, so it just flops around."

---

## Step 2: RUN EVOLUTION

1. In terminal (stop the visualizer first with Ctrl+C if needed), run:
   ```bash
   python run_genetic_algorithm.py --generations 2
   ```
   (Use 2 for a quick demo, or 5–10 for a longer run)

2. Let it run. Show the progress bar.

**What to say:** "Now I'm running the genetic algorithm with AFPO. Each generation, robots mutate, crossover, and the best ones survive."

---

## Step 3: FITNESS PLOT (Evolution visualization)

1. Open `plot_fitness.ipynb`
2. Run **Cell 1** (imports)
3. Run **Cell 2** (evolution overview: best, mean, and all individuals)
4. Run **Cell 4** (top 3 performers)

**What to say:** "This shows how fitness improved over generations. The red line is the best individual; the blue line is the population mean. The faded lines are individual robots. You can see the population getting better over time."

---

## Step 4: AFTER EVOLUTION — Show the evolved robot

### 4a. Morphology in the notebook

1. Open `visualize_robots.ipynb`
2. Run **Cell 1** (if not already run)
3. Run **Cell 6** (loads and plots robot_0, robot_1, robot_2)

**What to say:** "These are the top 3 evolved shapes. Compare to the random one."

### 4b. Evolved robot walking in the visualizer

1. In terminal, run:
   ```bash
   python visualizer.py --input robot_0.npy --port 5001
   ```
2. Open http://localhost:5001

**What to say:** "And here's the best one actually walking. It learned to move."

---

## Step 5: SIMPLE EXPLANATION (for the video)

Say something like:

> "I evolved virtual robots. I started with random shapes that couldn't move. I used a genetic algorithm: the robots that moved farther had more offspring. Over generations, the shapes got better. The best one now walks."

---

## Quick Reference — Commands in Order

| Step | Command | What it does |
|------|---------|--------------|
| 1b | `python save_random_robot.py` | Saves random robot |
| 1b | `python visualizer.py --input random_robot.npy --port 5001` | Shows random robot (poor movement) |
| 2 | `python run_genetic_algorithm.py --generations 5` | Runs evolution |
| 3 | Run `plot_fitness.ipynb` | Shows fitness over generations |
| 4b | `python visualizer.py --input robot_0.npy --port 5001` | Shows best evolved robot |

---

## Optional: Side-by-Side Comparison

Record two browser windows side by side:
- Left: random_robot.npy (flopping)
- Right: robot_0.npy (walking)

---

## Files You Need

- `visualize_robots.ipynb` — before/after morphology
- `plot_fitness.ipynb` — fitness plot
- `save_random_robot.py` — creates random robot for visualizer
- `run_genetic_algorithm.py` — evolution
- `visualizer.py` — shows robot moving
