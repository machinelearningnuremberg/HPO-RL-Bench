import numpy as np
import time
import torch
from scipy.stats import norm
from optimizers.gp_utils import get_model_likelihood_mll, train, propose_location
import numpy as np
import gpytorch
import logging
import itertools
from optimizers.optimizer import Optimizer


class GP(Optimizer):

    def __init__(self, search_space: dict, obj_function = None, max_budget: int = 100):
        super().__init__(search_space, obj_function)
        self.cartesian_prod_of_configurations = list(itertools.product(*tuple(search_space.values())))
        self.valid_configurations = [dict(zip(self.hp_names, x)) for x in self.cartesian_prod_of_configurations]
        self.observed_config = []
        self.pending_config = np.arange(len(self.valid_configurations)).tolist()
        self.constant_budget = max_budget
        self.backbone_params = {"kernel": "52", "nu": 0.5, "ard": True, "epochs": 30000, "loss_tol": 0.0001,
                                "dropout": 0.0, "lr": 0.01, "patience": 10, "dim": len(self.hp_names)}


    def suggest(self, n_iterations: int, init_configs: list = None):
        y = {}
        x = []
        incumbent = -np.inf
        if init_configs is not None:
            for conf in init_configs:
                result = self.obj_function(config=conf, budget=self.constant_budget)
                conf_ = []
                for hp in self.hp_names:
                    conf_.append(conf[hp])
                y[tuple(conf_)] = result["eval_avg_returns"][:self.constant_budget]
                x.append(tuple(conf_))
                if result["eval_avg_returns"][self.constant_budget-1] > incumbent:
                    incumbent = result["eval_avg_returns"][self.constant_budget-1]
                    inc_x = conf
            for _ in range(n_iterations):
                done = False
                if len(x) == len(self.valid_configurations):
                    break
                while not done:
                    model, likelihood, mll = get_model_likelihood_mll(1000, len(self.hp_names), self.backbone_params)
                    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': self.backbone_params["lr"]}])
                    model, likelihood, mll, max_Y = train(model, likelihood, mll, x, y, optimizer,
                                                          config=self.backbone_params)

                    candidate = propose_location(incumbent=incumbent, X_sample=x, Y_sample=y, gpr=model,
                                                 likelihood=likelihood,
                                                 config_space=self.valid_configurations,
                                                 hp_names=self.hp_names,
                                                 budget=self.constant_budget, dim=len(self.hp_names), inc_x=inc_x, max_Y=max_Y,
                                                 conf=self.backbone_params)
                    next_config = self.valid_configurations[candidate]
                    result = self.obj_function(config=next_config, budget=self.constant_budget)
                    conf_ = []
                    for hp in self.hp_names:
                        conf_.append(conf[hp])
                    y[tuple(conf_)] = result["eval_avg_returns"][:self.constant_budget]
                    x.append(tuple(conf_))
                    if result["eval_avg_returns"][self.constant_budget-1] > incumbent:
                        incumbent = result["eval_avg_returns"][self.constant_budget-1]
                        inc_x = next_config
                    done = True
                    print(incumbent)
        return inc_x, incumbent