import numpy as np
import itertools
from benchmark_handler import BenchmarkHandler
import matplotlib.pyplot as plt
from optimizers.pbt import PBT
from optimizers.pbt_pb2_utils import RunRLAlgorithm


search_space = "PPO"
benchmark = BenchmarkHandler(data_path='',
                             environment = "Pong-v0",
                             search_space = search_space,
                             return_names = ["returns_eval"],
                             seed = 0)

benchmark_ = RunRLAlgorithm(environment = "Pong-v0",
                           search_space = search_space,
                           seed = 0)

pbt_ = PBT(search_space_name=search_space, search_space=benchmark.get_search_space(search_space),
                             obj_function=benchmark_.ppo_cartpole,
                             max_budget=99, seed=0)

best_conf, best_score = pbt_.suggest()
print(f"Best configuration found is {best_conf}")
print(f"Best final evaluation return is {best_score}")



