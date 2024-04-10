from benchmark_handler import BenchmarkHandler
from optimizers.random_search import RandomSearch

search_space = "PPO"

benchmark = BenchmarkHandler(environment="Pong-v0",
                             search_space=search_space,
                             seed=0)

random_search = RandomSearch(search_space=benchmark.get_search_space(search_space),
                             obj_function=benchmark.get_metrics,
                             max_budget=99)

n_iters = 100
response_list = []
incumbents_list = []
best_conf, best_score = random_search.suggest(n_iters)
print(f"Best configuration found is {best_conf}")
print(f"Best final evaluation return is {best_score}")



