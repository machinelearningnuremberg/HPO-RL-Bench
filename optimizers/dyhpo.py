import argparse
import os
import random
import numpy as np
import itertools
from optimizers.optimizer import Optimizer
import numpy as np
from optimizers.hpo_method import DyHPOAlgorithm

class DyHPO(Optimizer):

    def __init__(self, search_space_name: str, search_space: dict, obj_function = None, max_budget: int = 100,
                 seed: int = 1, fantasize_step: int = 5, output_dir: str = "dyhpo", min_value: float = -np.inf,
                 max_value: float = np.inf, minimization: bool = False):
        super().__init__(search_space, obj_function)
        self.search_space_name = search_space_name
        self.seed = seed
        self.cartesian_prod_of_configurations = list(itertools.product(*tuple(search_space.values())))
        self.valid_configurations = [dict(zip(self.hp_names, x)) for x in self.cartesian_prod_of_configurations]
        self.observed_config = []
        self.pending_config = np.arange(len(self.valid_configurations)).tolist()
        self.constant_budget = max_budget
        self.fantasize_step = fantasize_step
        self.output_dir = output_dir
        self.min_value = min_value
        self.max_value = max_value
        self.log_indicator = [False, False, False, False, False]
        self.minimization = minimization
        os.makedirs(self.output_dir, exist_ok=True)


    def suggest(self, n_iterations: int = 1):
        budget_limit = self.constant_budget * n_iterations
        random.seed(self.seed)

        dyhpo_surrogate = DyHPOAlgorithm(
            np.array(self.cartesian_prod_of_configurations),
            self.log_indicator,
            seed=self.seed,
            max_benchmark_epochs=self.constant_budget,
            fantasize_step=self.fantasize_step,
            minimization=self.minimization,
            total_budget=budget_limit,
            dataset_name="%s_%s" % (self.search_space_name, self.seed),
            output_path=self.output_dir,
        )

        evaluated_configs = dict()
        method_budget = 0
        incumbent = -np.inf

        while method_budget < budget_limit:

            hp_index, budget = dyhpo_surrogate.suggest()
            config_tuple = np.array(self.cartesian_prod_of_configurations)[hp_index]
            config = {}
            for i, hp_name in enumerate(self.hp_names):
                config[hp_name] = config_tuple[i]
            result = self.obj_function(config=config, budget=budget)
            performance_curve = result["returns_eval"]
            score = performance_curve[-1]
            dyhpo_surrogate.observe(hp_index, budget, performance_curve)
            budget_cost = 0
            if hp_index in evaluated_configs:
                previous_budget = evaluated_configs[hp_index]
                budget_cost = budget - previous_budget
                evaluated_configs[hp_index] = budget
            else:
                budget_cost = self.fantasize_step

            method_budget += budget_cost

            if score > incumbent:
                incumbent = score
                inc_config = config

        return inc_config, incumbent