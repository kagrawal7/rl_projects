from time import perf_counter

import gymnasium as gym
import numpy as np


ACTIONS = ("Left", "Down", "Right", "Up")
REWARDS = {
    "sparse": (1.0, 0.0, 0.0),
    "dense": (1.0, -1.0, -0.01),
}


def make_env(reward="sparse", render_mode=None):
    return gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=True,
        render_mode=render_mode,
        reward_schedule=REWARDS[reward],
    )


class FrozenLakeDP:
    def __init__(self, env, gamma=0.9, theta=1e-3):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.P = env.unwrapped.P
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.terminals = {
            s for s, tile in enumerate(env.unwrapped.desc.reshape(-1))
            if tile in (b"H", b"G")
        }

    def q(self, state, action, values):
        total = 0
        for prob, next_state, reward, done in self.P[state][action]:
            total += prob * (reward + (0 if done else self.gamma * values[next_state]))
        return total

    def best_action(self, state, values):
        return int(np.argmax([self.q(state, a, values) for a in range(self.n_actions)]))

    def evaluate_policy(self, policy, values):
        sweeps = 0
        while True:
            sweeps += 1
            delta = 0
            new_values = values.copy()
            for s in range(self.n_states):
                if s in self.terminals:
                    continue
                new_values[s] = self.q(s, policy[s], values)
                delta = max(delta, abs(new_values[s] - values[s]))
            values = new_values
            if delta < self.theta:
                return values, sweeps

    def policy_iteration(self):
        start = perf_counter()
        values = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        iterations = sweeps = 0

        while True:
            values, new_sweeps = self.evaluate_policy(policy, values)
            sweeps += new_sweeps
            iterations += 1

            old_policy = policy.copy()
            for s in range(self.n_states):
                if s not in self.terminals:
                    policy[s] = self.best_action(s, values)
            if np.array_equal(policy, old_policy):
                break

        return {
            "algorithm": "Policy Iteration",
            "values": values,
            "policy": policy,
            "iterations": iterations,
            "sweeps": sweeps,
            "runtime": perf_counter() - start,
        }

    def value_iteration(self):
        start = perf_counter()
        values = np.zeros(self.n_states)
        iterations = 0

        while True:
            iterations += 1
            delta = 0
            new_values = values.copy()
            for s in range(self.n_states):
                if s in self.terminals:
                    continue
                new_values[s] = max(self.q(s, a, values) for a in range(self.n_actions))
                delta = max(delta, abs(new_values[s] - values[s]))
            values = new_values
            if delta < self.theta:
                break

        policy = np.array([
            0 if s in self.terminals else self.best_action(s, values)
            for s in range(self.n_states)
        ])
        return {
            "algorithm": "Value Iteration",
            "values": values,
            "policy": policy,
            "iterations": iterations,
            "sweeps": iterations,
            "runtime": perf_counter() - start,
        }


def run_experiments(gammas=(0.9, 0.99), rewards=("sparse", "dense"), episodes=100):
    experiments = []
    for algorithm in ("policy_iteration", "value_iteration"):
        theta = 1e-3 if algorithm == "policy_iteration" else 1e-6
        for reward in rewards:
            for gamma in gammas:
                solver = FrozenLakeDP(make_env(reward), gamma=gamma, theta=theta)
                result = getattr(solver, algorithm)()
                successes, path = 0, []
                for i in range(episodes):
                    reached_goal, path = play(result["policy"], reward, seed=10 + i)
                    successes += reached_goal
                result.update({
                    "reward": reward,
                    "gamma": gamma,
                    "theta": theta,
                    "success_rate": successes / episodes,
                    "latest_path": path,
                })
                experiments.append(result)
    return experiments


def play(policy, reward="sparse", seed=0, max_steps=100):
    env = make_env(reward)
    state, _ = env.reset(seed=seed)
    path = [int(state)]

    for _ in range(max_steps):
        state, _, done, truncated, _ = env.step(int(policy[state]))
        path.append(int(state))
        if done or truncated:
            break

    return path[-1] == 15, path


def plot_summary(experiments):
    import matplotlib.pyplot as plt

    labels = [f"{e['algorithm'][:2]}, {e['reward']}, g={e['gamma']}" for e in experiments]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    plots = [
        ("Runtime", [exp["runtime"] * 1000 for exp in experiments], "Milliseconds"),
        ("Evaluation Work", [exp["sweeps"] for exp in experiments], "Sweeps"),
        ("Rollout Success", [exp["success_rate"] for exp in experiments], "Rate"),
    ]
    for ax, (title, values, ylabel) in zip(axes, plots):
        ax.bar(labels, values)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=75)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()


def plot_values(experiments):
    import matplotlib.pyplot as plt

    states = np.arange(16)
    fig, axes = plt.subplots(len(experiments), 2, figsize=(10, 3 * len(experiments)))
    for ax_row, exp in zip(np.atleast_2d(axes), experiments):
        title = f"{exp['algorithm'][:2]}, {exp['reward']}, g={exp['gamma']}"
        ax_row[0].plot(states, exp["values"], marker="o")
        ax_row[0].set(title=title, xlabel="State", ylabel="V(s)")
        ax_row[0].grid(alpha=0.35)

        heatmap = ax_row[1].imshow(exp["values"].reshape(4, 4), cmap="viridis")
        ax_row[1].set(title=f"{title} heatmap", xticks=[], yticks=[])
        fig.colorbar(heatmap, ax=ax_row[1], fraction=0.046, pad=0.04)
    fig.tight_layout()


def named_policy(policy):
    return np.array([ACTIONS[a] for a in policy]).reshape(4, 4)
