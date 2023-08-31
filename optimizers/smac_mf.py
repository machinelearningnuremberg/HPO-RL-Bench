import numpy as np
import itertools
from optimizers.optimizer import Optimizer
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

class SMAC_MF(Optimizer):

    def __init__(self, search_space: dict, min_budget: int = 5, max_budget: int = 100, eta : int = 3, seed: int = 0,
                 obj_function = None):
        super().__init__(search_space, obj_function)
        self.cartesian_prod_of_configurations = list(itertools.product(*tuple(search_space.values())))
        self.valid_configurations = [dict(zip(self.hp_names, x)) for x in self.cartesian_prod_of_configurations]
        self.observed_config = []
        self.observed_y = []
        self.pending_config = np.arange(len(self.valid_configurations)).tolist()
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.cs = self.get_configspace()
        self.seed = seed
        self.intensifier_kwargs = {"initial_budget": self.min_budget, "max_budget": self.max_budget, "eta": 3}

    def get_configspace(self):
        cs = ConfigurationSpace()
        for hp_name, values in self.search_space.items():
                cs.add_hyperparameter(CategoricalHyperparameter(hp_name, choices=values))
        return cs

    def suggest(self, n_iterations: int):
        cs = self.get_configspace()
        # SMAC scenario object
        scenario = Scenario(
            {
                "run_obj": "quality",
                "runcount-limit": n_iterations,
                "cs": cs,
                "deterministic": True,
                "limit_resources": False,
            }
        )
        smac = SMAC4MF(
            scenario=scenario,
            rng=np.random.RandomState(self.seed),
            tae_runner=self.obj_function,
            intensifier_kwargs=self.intensifier_kwargs,
        )
        # Start optimization
        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent
        inc_value = self.obj_function(config=incumbent, budget=self.max_budget)

        return dict(incumbent), inc_value
