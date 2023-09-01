import numpy as np
import itertools
from benchmark_handler import BenchmarkHandler
import matplotlib.pyplot as plt
from optimizers.gp import GP



search_space = "PPO"

benchmark = BenchmarkHandler(data_path='',
                             environment = "Pong-v0",
                             search_space = search_space,
                             return_names = ["eval_avg_returns"],
                             seed = 0)

gp = GP(search_space=benchmark.get_search_space(search_space),
                             obj_function=benchmark.get_metrics,
                             max_budget=99)

n_iters = 10
n_init_configs = 4
init_configs_idx = np.random.choice(gp.pending_config, n_init_configs)
init_configs = []
for idx in init_configs_idx:
    init_configs.append(gp.valid_configurations[idx])
best_conf, best_score = gp.suggest(n_iterations=n_iters, init_configs=init_configs)
print(f"Best configuration found is {best_conf}")
print(f"Best final evaluation return is {best_score}")



