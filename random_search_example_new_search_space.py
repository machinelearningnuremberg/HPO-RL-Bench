import numpy as np
import itertools
from benchmark_handler import BenchmarkHandler
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from optimizers.random_search import RandomSearch


search_space = {"DQN":
                    {"learning_rate": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                     "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]}
                }

benchmark = BenchmarkHandler(environment='CartPole-v1',
                             search_space=search_space,
                             seed=0,
                             rl_algorithm=DQN)

random_search = RandomSearch(search_space=benchmark.get_search_space(search_space), obj_function=benchmark.get_metrics)

n_iters = 2
response_list = []
incumbents_list = []
best_conf, best_score = random_search.suggest(n_iters)
print(f"Best configuration found is {best_conf}")
print(f"Best final evaluation return is {best_score}")
