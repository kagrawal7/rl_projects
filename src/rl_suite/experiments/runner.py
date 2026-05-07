from time import perf_counter

import gymnasium as gym
import numpy as np


class GymExperiment:
    def __init__(self, env, algorithm, env_kwargs=None, train=None, act=None):
        self.env = env
        self.algorithm = algorithm
        self.env_kwargs = env_kwargs or {}
        self.train = train
        self.act = act

    def make_env(self, seed=None, **kwargs):
        env_kwargs = {**self.env_kwargs, **kwargs}
        if isinstance(self.env, str):
            env = gym.make(self.env, **env_kwargs)
        elif callable(self.env):
            env = self.env(**env_kwargs)
        else:
            env = self.env
        if seed is not None:
            env.reset(seed=seed)
        return env

    def make_agent(self, env, config):
        if isinstance(self.algorithm, type):
            return self.algorithm(env, **config)
        return self.algorithm(env, **config)

    def run(self, configs=None, eval_episodes=20, seed=0):
        rows = []
        for raw_config in configs or [{}]:
            config = dict(raw_config)
            env_config = config.pop("env", {})
            agent_config = config.pop("agent", config)
            env = self.make_env(seed=seed, **env_config)
            agent = self.make_agent(env, agent_config)

            start = perf_counter()
            history = train_agent(agent, env, self.train)
            train_time = perf_counter() - start

            scores = evaluate(
                agent,
                lambda episode_seed: self.make_env(seed=episode_seed, **env_config),
                episodes=eval_episodes,
                seed=seed,
                act=self.act,
            )
            rows.append({
                "config": raw_config,
                "agent": agent,
                "history": history,
                "train_time": train_time,
                **scores,
            })
            env.close()
        return rows


def train_agent(agent, env, train=None):
    if train is not None:
        return train(agent, env)
    for name in ("train", "learn", "fit", "control"):
        if hasattr(agent, name):
            return getattr(agent, name)(env)
    return None


def evaluate(agent, env_factory, episodes=20, seed=0, act=None, max_steps=1000):
    rewards, lengths, last_path = [], [], []
    successes = 0

    for i in range(episodes):
        env = env_factory(seed + i)
        reward, steps, done, path = play(env, agent, seed + i, act, max_steps)
        rewards.append(reward)
        lengths.append(steps)
        last_path = path
        successes += done
        env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": successes / episodes,
        "last_path": last_path,
    }


def play(env, agent, seed=0, act=None, max_steps=1000):
    state, _ = env.reset(seed=seed)
    path = [state]
    total_reward = 0

    for step in range(1, max_steps + 1):
        action = choose_action(agent, state, act)
        state, reward, terminated, truncated, _ = env.step(action)
        path.append(state)
        total_reward += reward
        if terminated or truncated:
            return total_reward, step, terminated, path

    return total_reward, max_steps, False, path


def choose_action(agent, state, act=None):
    if act is not None:
        return act(agent, state)
    if hasattr(agent, "select_action"):
        return agent.select_action(state)
    if hasattr(agent, "act"):
        return agent.act(state)
    if hasattr(agent, "predict"):
        action = agent.predict(state)
        return action[0] if isinstance(action, tuple) else action
    if callable(agent):
        return agent(state)
    raise TypeError("Pass an agent with select_action/act/predict or provide act=...")
