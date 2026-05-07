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
|       |-- algorithms.py
|       |-- callbacks/
|       |-- environments/
|       |-- experiments/
|       |-- visualization/
|       |-- utils/
|       |-- dp.py
|       |-- mc.py
|       |-- td.py
|       |-- _base.py
|       `-- _utilities/
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

The public API is organized around algorithms, environments, experiments, and
visualization helpers.

```python
import gymnasium as gym
from rl_suite.environments import run_episode
from rl_suite.visualization import render_env_in_notebook

env = gym.make("FrozenLake-v1", render_mode="rgb_array")

run_episode(env, render_fn=render_env_in_notebook, render_each_step=True)
```

Algorithms can be imported directly:

```python
from rl_suite.algorithms import QLearning, SARSA, ExpectedSARSA, ValueIteration

q_learning_agent = QLearning(env)
sarsa_agent = SARSA(env)
value_iteration_agent = ValueIteration(env)
```

Algorithm families are also available as discoverable modules:

```python
from rl_suite import td

print(td.get_implemented_algorithms())
# ['QLearning', 'SARSA', 'ExpectedSARSA']

agent = td.QLearning(env)
```

Experiment callbacks are available from the package-level callbacks module:

```python
from rl_suite.callbacks import CallbackList, RolloutCallback
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

- Public algorithm classes live in `src/rl_suite/algorithms.py` and the family
  modules `src/rl_suite/td.py`, `src/rl_suite/mc.py`, and `src/rl_suite/dp.py`.
- Public callback hooks live under `src/rl_suite/callbacks`.
- Public environment helpers live under `src/rl_suite/environments`.
- Public experiment helpers live under `src/rl_suite/experiments`.
- Shared base classes live in `src/rl_suite/_base.py`; utility implementation
  details remain under `src/rl_suite/_utilities`.
