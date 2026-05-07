# RL Projects

This repository contains reinforcement learning experiments that grew out of
assignments from two different courses. The original work was notebook-based;
the repeated environment execution, discretization, rendering, and algorithm
code has since been refactored into a small reusable Python package.

All project code is my own. AI assistance was used during the refactor from
notebooks into a package.

## Project Structure

```text
.
|-- notebooks/
|   |-- frozen_lake.ipynb
|   |-- cartpole_mc.ipynb
|   `-- cartpole_td.ipynb
|-- src/
|   `-- rl_suite/
|       |-- RLToolbox/
|       |-- _utilities/
|       `-- _algorithms/
|-- requirements.txt
`-- pyproject.toml
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e .
```

For notebook work, install the notebook extras:

```bash
pip install -e ".[notebooks]"
```

## Usage

The main public entry point is `rl_suite.RLToolbox`.

```python
import gymnasium as gym
import rl_suite.RLToolbox.utilities as utils

env = gym.make("FrozenLake-v1", render_mode="rgb_array")

render_fn = utils.render_env_in_notebook

utils.run_episode(env, render_fn=render_fn, render_each_step=True)
```

Algorithms are accessed through the same toolbox facade:

```python
import rl_suite.RLToolbox.algorithms as rl

sarsa_agent = rl.td.sarsa(env)
q_learning_agent = rl.td.q_learning(env)
mc_agent = rl.mc.off_policy(env)
```

## Notebooks

The notebooks demonstrate the package APIs without reimplementing environment
execution or algorithms inline.

- `notebooks/frozen_lake.ipynb`: FrozenLake execution, rendering, and dynamic
  programming examples.
- `notebooks/cartpole_mc.ipynb`: CartPole with Monte Carlo control and
  discretized observation spaces.
- `notebooks/cartpole_td.ipynb`: CartPole with temporal-difference control
  algorithms.

Start Jupyter with:

```bash
jupyter lab
```

## Development Notes

- Utility code lives in `src/rl_suite/_utilities`.
- Algorithm implementations live in `src/rl_suite/_algorithms`.
- The notebooks should use `rl_suite.RLToolbox` as the public interface rather
  than duplicating algorithm or environment execution code.
