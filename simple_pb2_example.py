from benchmark_handler import BenchmarkHandler
from optimizers.pb2 import PB2Optimizer
from optimizers.pbt_pb2_utils import RunRLAlgorithm


search_space = "PPO"
benchmark = BenchmarkHandler(environment = "CartPole-v1",
                             search_space = search_space,
                             seed = 0)

benchmark_ = RunRLAlgorithm(environment = "CartPole-v1",
                           search_space = search_space,
                           seed = 0)

pb2_ = PB2Optimizer(search_space_name=search_space, search_space=benchmark.get_search_space(search_space),
                    obj_function=benchmark_.ppo_cartpole,
                    max_budget=99, seed=0)

best_conf, best_score = pb2_.suggest()
print(f"Best configuration found is {best_conf}")
print(f"Best final evaluation return is {best_score}")



