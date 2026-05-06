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
|       |-- rl_toolbox.py
|       |-- utils/
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
import rl_suite as rl

env = gym.make("FrozenLake-v1", render_mode="rgb_array")

utils = rl.RLToolbox.utils
runner = utils.RLEnvironmentRunner(env)
render_fn = utils.render_env_in_notebook
```

For environment-bound utilities:

```python
toolbox = rl.RLToolbox(env)

human_runner = toolbox.utils.human_agent
rl_runner = toolbox.utils.rl_agent
```

Algorithms are accessed through the package factory:

```python
agent = rl.algorithms(env)

sarsa_agent = agent.td.sarsa
q_learning_agent = agent.td.q_learning
mc_agent = agent.mc.off_policy
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

- Utility code lives in `src/rl_suite/utils`.
- Algorithm implementations live in `src/rl_suite/_algorithms`.
- The notebooks should use `rl_suite` as the public interface rather than
  duplicating algorithm or environment execution code.
