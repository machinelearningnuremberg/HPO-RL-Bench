import numpy as np
import itertools
from optimizers.optimizer import Optimizer


class RandomSearch(Optimizer):

    def __init__(self, search_space: dict, obj_function = None, max_budget: int = 100):
        super().__init__(search_space, obj_function)
        self.cartesian_prod_of_configurations = list(itertools.product(*tuple(search_space.values())))
        self.valid_configurations = [dict(zip(self.hp_names, x)) for x in self.cartesian_prod_of_configurations]
        self.observed_config = []
        self.pending_config = np.arange(len(self.valid_configurations)).tolist()
        self.constant_budget = max_budget

    def suggest(self, n_iterations: int = 1):
        next_confs_ix = np.random.choice(self.pending_config, n_iterations)
        best_return = -np.inf
        best_conf = None
        for i in range(n_iterations):
            conf_ix = next_confs_ix[i]
            conf = self.valid_configurations[conf_ix]
            result = self.obj_function(config=conf, budget=self.constant_budget)
            final_avg_reward = result["returns_eval"][self.constant_budget-1]
            if final_avg_reward >= best_return:
                best_return = final_avg_reward
                best_conf = conf




        return best_conf, best_return
