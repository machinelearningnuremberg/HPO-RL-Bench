import numpy as np
import itertools
from benchmark_handler import BenchmarkHandler
import matplotlib.pyplot as plt

class RandomSearch:

    def __init__(self, search_space : dict):

        self.search_space = search_space

        self.cartesian_prod_of_configurations = list(itertools.product(*tuple(search_space.values())))
        self.hp_names =  list(search_space.keys())
        self.valid_configurations = [dict(zip(self.hp_names, x)) for x in self.cartesian_prod_of_configurations]
        self.observed_config = []
        self.pending_config =  np.arange(len(self.valid_configurations)).tolist()
        self.constant_budget = 100

    
    def observe_and_suggest(self, conf: dict):

        if conf != {}:
            ix = self.cartesian_prod_of_configurations.index(tuple(conf.values()))
            self.pending_config.remove(ix)
            self.observed_config.append(ix)

        next_conf_ix = np.random.choice(self.pending_config, 1).item()

        return self.valid_configurations[next_conf_ix]

search_space = "PPO"

benchmark = BenchmarkHandler(data_path='',
                             environment = "Pong-v0",
                             search_space = search_space,
                             return_names = ["returns_eval"],
                             seed = 0)

random_search = RandomSearch(benchmark.get_search_space(search_space))

n_iters = 100
response_list = []
incumbents_list = []

for i in range(n_iters):

    if i == 0:
        next_conf = random_search.observe_and_suggest({})
    else:
        next_conf = random_search.observe_and_suggest(next_conf)
    
    metrics = benchmark.get_metrics(next_conf, budget = 99) #full budget
    response_list.append(metrics["returns_eval"][-1])
    incumbents_list.append(max(response_list))

plt.plot(incumbents_list)



