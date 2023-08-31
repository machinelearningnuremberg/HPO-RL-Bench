import itertools
from optimizers.optimizer import Optimizer
import numpy as np
import os
import optuna
import yaml
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from optuna.pruners import BasePruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optimizers.optuna_hyperparams import HYPERPARAMS_SAMPLER
import pickle as pkl
class Optuna(Optimizer):

    def __init__(self, search_space_name: str, search_space: dict, max_budget: int = 100, budget_increments: int = 10,
                 seed: int = 0, obj_function = None, sampler="tpe", pruner="median", n_startup_trials=4, n_jobs=1,
                 log_folder="results"):
        super().__init__(search_space, obj_function)
        self.search_space_name = search_space_name
        self.cartesian_prod_of_configurations = list(itertools.product(*tuple(search_space.values())))
        self.valid_configurations = [dict(zip(self.hp_names, x)) for x in self.cartesian_prod_of_configurations]
        self.observed_config = []
        self.observed_y = []
        self.pending_config = np.arange(len(self.valid_configurations)).tolist()
        self.constant_budget = max_budget
        self.budget_inc = budget_increments
        self.seed = seed
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_jobs = n_jobs
        self.log_folder = log_folder
        self.log_path = f"{log_folder}/{search_space_name}/{seed}"
        self.save_path = os.path.join(
            self.log_path, f"{search_space_name}_{seed}")
        self.params_path = f"{self.save_path}/{search_space_name}"

    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
            """
            Save unprocessed hyperparameters, this can be use later
            to reproduce an experiment.

            :param saved_hyperparams:
            """
            # Save hyperparams
            with open(os.path.join(self.params_path, "config.yml"), "w") as f:
                yaml.dump(saved_hyperparams, f)

    def create_log_folder(self):
        os.makedirs(self.params_path, exist_ok=True)

    def _create_sampler(self) -> BaseSampler:
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if self.sampler == "random":
            sampler = RandomSampler(seed=self.seed)
        elif self.sampler == "tpe":
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed, multivariate=True)
        elif self.sampler == "skopt":
            from optuna.integration.skopt import SkoptSampler

            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")
        return sampler

    def _create_pruner(self) -> BasePruner:
        if self.pruner == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif self.pruner == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_startup_trials+5)
        elif self.pruner == "none":
            # Do not prune
            pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.pruner}")
        return pruner

    def objective(self, trial: optuna.Trial) -> float:

        # kwargs = self._hyperparams.copy()

        # Hack to use DDPG/TD3 noise sampler
        # trial.n_actions = n_actions
        # Hack when using HerReplayBuffer
        # trial.using_her_replay_buffer = kwargs.get("replay_buffer_class") == HerReplayBuffer
        # if trial.using_her_replay_buffer:
        #     trial.her_kwargs = kwargs.get("replay_buffer_kwargs", {})
        # Sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.search_space_name](trial)
        # kwargs.update(sampled_hyperparams)

        try:
            config = dict()
            for hp_name in self.hp_names:
                config[hp_name] = sampled_hyperparams[hp_name]
            for i, bud in enumerate(np.arange(self.budget_inc, self.constant_budget, self.budget_inc)):
                result = self.obj_function(config=config, budget=bud)
                final_return = result["returns_eval"][-1]
                trial.report(final_return, i)
                if trial.should_prune():
                    break
        except (AssertionError, ValueError) as e:
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            print(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()

        return final_return

    def suggest(self, n_iterations: int = 1):

        sampler = self._create_sampler()
        pruner = self._create_pruner()

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=None,
            study_name=f"{self.search_space_name}_{self.seed}",
            load_if_exists=True,
            direction="maximize",
        )

        try:
            study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=n_iterations)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.search_space_name}_{n_iterations}_trials_{int(self.seed)}"
        )

        log_path = os.path.join(self.log_folder, self.search_space_name, report_name)

        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f"{log_path}.csv")

        # Save python object to inspect/re-use it later
        with open(f"{log_path}.pkl", "wb+") as f:
            pkl.dump(study, f)

        return trial.params, trial.value