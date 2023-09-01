import numpy as np
import itertools
from benchmark_handler import BenchmarkHandler
import matplotlib.pyplot as plt
from optimizers.smac_mf import SMAC_MF
from functools import partial


search_space = "PPO"

benchmark = BenchmarkHandler(data_path='',
                             environment = "Pong-v0",
                             search_space = search_space,
                             return_names = ["eval_avg_returns"],
                             seed = 0)

gp = SMAC_MF(search_space=benchmark.get_search_space(search_space),
                             obj_function=partial(benchmark.get_metrics, return_final_only=True))

n_iters = 10
best_conf, best_score = gp.suggest(n_iterations=n_iters)
print(f"Best configuration found is {best_conf}")
print(f"Best final evaluation return is {best_score}")



