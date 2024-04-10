import numpy as np
from benchmark_handler import BenchmarkHandler
import matplotlib.pyplot as plt
from optimizers.optimizer import Optimizer

class RandomSearch(Optimizer):

    def __init__(self, max_budget=99, verbose=False):
        self.max_budget = max_budget
        self.verbose = verbose

    def observe_and_suggest(self, configurations, observed_lc):

        if self.verbose:
            print("Observed Learning Curves:")
            print(observed_lc)
        
        min_budget = self.max_budget+1
        while min_budget == (self.max_budget+1): #we iterate till we find a curve that is not complete
            ix = np.random.randint(len(configurations))
            if ix in observed_lc.keys():
                min_budget = len(observed_lc[ix])
            else:
                min_budget = 1
            budget = np.random.randint(min_budget, self.max_budget)
        return ix, budget


benchmark = BenchmarkHandler(environment = "Pong-v0",
                             search_space = "PPO",
                             seed = 0)




random_searcher = RandomSearch(verbose=False)
observed_lc, max_observed_perf, best_configuration, best_configuration_ix = benchmark.run_bo(optimizer=random_searcher,
                                                                                             iterations=1000)
plt.plot(observed_lc[best_configuration_ix])
plt.title("Best observed learning curve")
plt.show()

for lc in observed_lc.values():
    plt.plot(lc)
plt.title("All observed learning curves")
plt.show()