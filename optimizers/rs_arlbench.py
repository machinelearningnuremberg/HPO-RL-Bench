#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import json
import pickle
from autorlbench_utils import get_metrics
ENVS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
        'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
        'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
        'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
        'Bowling-v0', 'Asteroids-v0',
        'Ant-v2', 'Hopper-v2', 'Humanoid-v2',
        'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']

parser = argparse.ArgumentParser()
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--space', help='Gym Environment', type=int, default=1)
parser.add_argument('--algorithm', help='RL Algorithm', type=str, choices=["PPO", "DDPG", "A2C"], default="PPO")
parser.add_argument('--seed', help='Which seed', type=int, choices=[0, 1, 2], default=1)
args = parser.parse_args()

spaces = {"PPO": None,
          "DDPG": None,
          "A2C": None}
with open('config_space_ppo','rb') as f:
    spaces["PPO"] = pickle.load(f)
    if args.algorithm == "PPO":
        init_config = (-4, 0.8, 0.2)

with open('config_space_ddpg','rb') as f:
    spaces["DDPG"] = pickle.load(f)
    if args.algorithm == "DDPG":
        init_config = (-4, 0.8, 0.0001)

with open('config_space_a2c','rb') as f:
    spaces["A2C"] = pickle.load(f)
    if args.algorithm == "A2C":
        init_config = (-4, 0.8)

args.seed = int(args.seed)
config_space = spaces[args.algorithm]


n_init = 1
idxs = config_space.index(init_config)
x = [init_config]
y = []
X_support = []
print(x)
full_budget = 100
incumbent = -np.inf
start_time = time.time()
for cfg in x:
    config = None
    if args.algorithm == "PPO":
        config = {
            "lr": cfg[0],
            "gamma": cfg[1],
            "clip": cfg[2]
        }
    elif args.algorithm == "DDPG":
        config = {
            "lr": cfg[0],
            "gamma": cfg[1],
            "tau": cfg[2]
        }
    elif args.algorithm == "A2C":
        config = {
            "lr": cfg[0],
            "gamma": cfg[1]
        }
    results = get_metrics(search_space=args.algorithm, environment=ENVS[int(args.space)], config=config,
                          budget=full_budget, seed=args.seed)
    if results["eval_avg_returns"][-1] > incumbent:
        incumbent = results["eval_avg_returns"][-1]
        inc_x = cfg
        inc_budget = full_budget

    budget = full_budget
    run_data = {"incumbent": incumbent, "configuraton": inc_x,
                "wallclock_time": float(time.time() - start_time),
               "wallclock_time_eval": results["eval_timestamps"][-1], "eval_timesteps": results["eval_timesteps"][-1]
    }
    with open("RS_seed%s_%s_%s.json" % (args.seed, ENVS[args.space], args.algorithm), "a+") as f:
        json.dump(run_data, f)
candidates = np.random.randint(low=0, high=len(config_space), size=500)
for i in range(args.n_iters):

    candidate = candidates[i]
    if candidate in idxs:
        old_conf= True
        while old_conf:
            candidate = np.random.randint(low=0, high=len(config_space))
            if candidate not in idxs:
                old_conf = False
                break

    config = None
    cfg = config_space[candidate]
    if args.algorithm == "PPO":
        config = {
            "lr": int(config_space[candidate][0]),
            "gamma": config_space[candidate][1],
            "clip": config_space[candidate][2],
        }
    elif args.algorithm == "DDPG":
        config = {
            "lr": int(config_space[candidate][0]),
            "gamma": config_space[candidate][1],
            "tau": config_space[candidate][2],
        }
    elif args.algorithm == "A2C":
        config = {
            "lr": int(config_space[candidate][0]),
            "gamma": config_space[candidate][1],
        }
    print("Evaluating at: %s" % config)
    results = get_metrics(search_space=args.algorithm, environment=ENVS[int(args.space)], config=config,
                          budget=full_budget, seed=args.seed)
    y.append(results["eval_avg_returns"][-1])
    print("Result: %s" % y)
    X_support.append(config)
    print(len(X_support))
    if results["eval_avg_returns"][-1] > incumbent:
        incumbent = results["eval_avg_returns"][-1]
        inc_x = cfg
        inc_budget = full_budget
    print("Incumbent at %s" % (time.time() - start_time))
    print("Config: %s  Reward: %s" % (inc_x, incumbent))
    run_data = {"incumbent": incumbent, "configuraton": inc_x,
                "wallclock_time": float(time.time() - start_time),
               "wallclock_time_eval": results["eval_timestamps"][-1], "eval_timesteps": results["eval_timesteps"][-1]
    }
    with open("RS_seed%s_%s_%s.json" % (args.seed, ENVS[args.space], args.algorithm), "a+") as f:
        f.write('\n')
        json.dump(run_data, f)

print("Max budget: %s" % budget)
print("Incumbent")
print("Config: %s    Budget: %s    Reward: %s" % (inc_x, inc_budget, incumbent))