import json
import os
import numpy as np
import itertools
from typing import Union
from glob import glob

try:
    from utils import run_rl_algorithm
except Exception:
    print("utils not found, cannot run RL algorithm")


class BenchmarkHandler:

    def __init__(self, data_path: str = "", environment: str = None, search_space: Union[str, dict] = None,
                 set: str = "static", return_metrics=None, seed=0, rl_algorithm=None):
        """
        A handler for interacting with and running HPO-RL-Bench.
        """

        self.data_path = data_path

        self.environment = environment
        if isinstance(search_space, str) or search_space is None:
            self.search_space = search_space
        else:
            for ss_name, ss_dict in search_space.items():
                self.search_space = ss_name
                self.search_space_dict = ss_dict
            assert rl_algorithm is not None, "For new search spaces, an RL Algorithm object must be provided"
        self.rl_algorithm = rl_algorithm
        self.seed = seed
        self.return_metrics = return_metrics
        self.set = set
        self.search_space_structure = {
            "static": {
                "PPO": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "clip": [0.1, 0.2, 0.3]},
                "DQN": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "epsilon": [0.1, 0.2, 0.3]},
                "A2C": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]},
                "DDPG": {"lr": [-6, -5, -4, -3, -2, -1],
                         "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                         "tau": [0.01, 0.001, 0.005]},
                "SAC": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "tau": [0.01, 0.001, 0.005]},
                "TD3": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "tau": [0.01, 0.001, 0.005]}
            },
            "dynamic": {
                "PPO": {"lr": [-5, -4, -3],
                        "gamma": [0.95, 0.98, 0.99]}
            },
            "extended": {
                "PPO": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "clip": [0.1, 0.2, 0.3],
                        "n_layers": [1, 2, 3],
                        "n_units": [32, 64, 128, 256]},
                "DQN": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "epsilon": [0.1, 0.2, 0.3],
                        "n_layers": [1, 2, 3],
                        "n_units": [32, 64, 128, 256]},
                "A2C": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "n_layers": [1, 2, 3],
                        "n_units": [32, 64, 128, 256]},
                "DDPG": {"lr": [-6, -5, -4, -3, -2, -1],
                         "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                         "tau": [0.01, 0.001, 0.005],
                         "n_layers": [1, 2, 3],
                         "n_units": [32, 64, 128]},
                "SAC": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "tau": [0.01, 0.001, 0.005],
                        "n_layers": [1, 2, 3],
                        "n_units": [32, 64, 128]},
                "TD3": {"lr": [-6, -5, -4, -3, -2, -1],
                        "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                        "tau": [0.01, 0.001, 0.005],
                        "n_layers": [1, 2, 3],
                        "n_units": [32, 64, 128]}
            }
        }
        self.environment_dict = {
            "Atari": ["Pong-v0", "Alien-v0", "BankHeist-v0", "BeamRider-v0", "Breakout-v0", "Enduro-v0", "Phoenix-v0",
                      "Seaquest-v0", "SpaceInvaders-v0", "Riverraid-v0", "Tennis-v0", "Skiing-v0", "Boxing-v0",
                      "Bowling-v0", "Asteroids-v0"],
            "Mujoco": ["Ant-v2", "Hopper-v2", "Humanoid-v2"],
            "Control": ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0"]
        }

        self.environment_list = ["Pong-v0", "Ant-v2", "Alien-v0", "BankHeist-v0", "BeamRider-v0", "Breakout-v0",
                                 "Enduro-v0", "Phoenix-v0", "Seaquest-v0", "SpaceInvaders-v0", "Riverraid-v0",
                                 "Tennis-v0", "Skiing-v0", "Boxing-v0", "Bowling-v0", "Asteroids-v0", "Hopper-v2",
                                 "Humanoid-v2", "CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0"]
        self.seeds_list = [0, 1, 2, 3, 4]
        self.env_types = ["Atari", "Mujoco", "Control"]
        self.valid_dynamic_spaces = ["PPO"]
        self.valid_extended_spaces = ["PPO", "A2C", "DDPG", "SAC", "TD3", "DQN"]

        if self.set == "extended":
            self.environment_list = self.environment_dict["Control"] + self.environment_dict["Mujoco"]

        if self.return_metrics is None:
            self.return_metrics = ["eval_avg_returns", "eval_std_returns", "eval_timestamps", "eval_timesteps"]

        if self.search_space is not None:
            self._precompute_configurations()

    def set_env_space_seed(self, search_space: str, environment: str, seed: int):
        self.search_space = search_space
        self.environment = environment
        self.seed = seed
        self._precompute_configurations()

    def _build_return_dict(self, data, budget):
        max_budget_allowed = len(data["eval_timesteps"])
        assert budget <= max_budget_allowed, f"Budget should be lower than {max_budget_allowed}"

        if self.return_metrics is None:
            return data
        else:
            return {k: v for k, v in data.items() if k in self.return_metrics}

    def get_environments_groups(self):
        return list(self.environment_dict.keys())

    def get_environments_per_group(self, env_group: str = "Atari"):
        return self.environment_dict[env_group]

    def get_environments(self):
        return self.environment_list

    def get_search_spaces_names(self, set: str = "static"):
        return list(self.search_space_structure[set].keys())

    def get_metrics(
        self,
        config: dict,
        search_space: str = "",
        environment: str = "",
        seed: int = np.inf,
        budget: int = 100,
        set: str = "",
        return_final_only: bool = False,
    ):
        """
        Retrieves performance metrics for a given configuration.
        """

        def _resolve_in_shards(dirpath: str, filename: str) -> str:
            p = os.path.join(dirpath, filename)
            if os.path.exists(p):
                return p
            matches = glob(os.path.join(dirpath, "**", filename), recursive=True)
            if matches:
                return matches[0]
            raise FileNotFoundError(p)

        def fmt(x):
            """Format floats for filenames to match benchmark style."""
            if isinstance(x, float):
                s = f"{x:.6f}".rstrip("0").rstrip(".")
                if "." not in s:
                    s += ".0"
                return s
            return str(x)

        budget = int(budget)

        def _final_or_dict(data):
            # Pad lists to >= budget length
            def _pad_to(lst, target, filler=None):
                if len(lst) >= target:
                    return lst
                fill = lst[-1] if (filler is None and len(lst) > 0) else filler
                return lst + [fill] * (target - len(lst))

            data["timesteps_eval"] = _pad_to(data.get("timesteps_eval", []), budget,
                                             (len(data.get("timesteps_eval", [])) + 1) * 10000)
            data["returns_eval"] = _pad_to(data.get("returns_eval", []), budget)
            data["std_returns_eval"] = _pad_to(data.get("std_returns_eval", []), budget, 0.0)
            data["timestamps_eval"] = _pad_to(data.get("timestamps_eval", []), budget, 0)

            if not return_final_only:
                return self._build_return_dict(
                    data={
                        "eval_avg_returns": data["returns_eval"][:budget],
                        "eval_std_returns": data["std_returns_eval"][:budget],
                        "eval_timestamps": data["timestamps_eval"][:budget],
                        "eval_timesteps": data["timesteps_eval"][:budget],
                    },
                    budget=budget,
                )
            else:
                return data["returns_eval"][budget - 1]

        if self.rl_algorithm is None:
            if search_space == "":
                search_space = self.search_space
            if environment == "":
                environment = self.environment
            if seed is np.inf:
                seed = self.seed
            if set == "":
                set = self.set

            # 3-way shard by seed: 0-2 -> 0, 3-5 -> 1, 6-9 -> 2
            s = int(seed)
            if 0 <= s <= 2:
                part = 0
            elif 3 <= s <= 5:
                part = 1
            elif 6 <= s <= 9:
                part = 2
            else:
                raise ValueError(f"Unsupported seed {seed}, expected 0..9")

            def env_dir_for(space: str) -> str:
                if space in {"DQN", "PPO", "SAC"}:
                    return f"{environment}_{part}"
                return environment

            if set in ["static", "extended"]:
                lr = int(config.get("lr"))
                gamma = config.get("gamma")

                if search_space == "DQN":
                    epsilon = config.get("epsilon")
                    dirpath = os.path.join(self.data_path, "data_hpo_rl_bench", search_space, env_dir_for("DQN"))

                    if set == "static":
                        if environment in self.environment_dict["Atari"]:
                            fname = (f"{environment}_{search_space}_random_lr_{lr}"
                                     f"_gamma_{fmt(gamma)}_epsilon_{fmt(epsilon)}_seed{seed}_eval.json")
                        else:
                            n_layers = 2
                            n_units = 64
                            fname = (f"{environment}_{search_space}_lr_{lr}_gamma_{fmt(gamma)}"
                                    f"_epsilon_{fmt(epsilon)}_layers_{n_layers}_units_{n_units}_seed{seed}.json")

                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        fname = (f"{environment}_{search_space}_lr_{lr}_gamma_{fmt(gamma)}"
                                 f"_epsilon_{fmt(epsilon)}_layers_{n_layers}_units_{n_units}_seed{seed}.json")

                    p = _resolve_in_shards(dirpath, fname)
                    with open(p) as f:
                        data = json.load(f)
                    return _final_or_dict(data)

                elif search_space == "PPO":
                    clip = config.get("clip")
                    dirpath = os.path.join(self.data_path, "data_hpo_rl_bench", search_space, env_dir_for("PPO"))

                    if set == "static":
                        if environment in self.environment_dict["Atari"]:
                            fname = (f"{environment}_{search_space}_random_lr_{lr}"
                                    f"_gamma_{fmt(gamma)}_clip_{fmt(clip)}_seed{seed}_eval.json")
                        else:
                            n_layers = 2
                            n_units = 64
                            fname = (f"{environment}_{search_space}_lr_{lr}_gamma_{fmt(gamma)}"
                                    f"_clip_{fmt(clip)}_layers_{n_layers}_units_{n_units}_seed{seed}.json")
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        fname = (f"{environment}_{search_space}_lr_{lr}_gamma_{fmt(gamma)}"
                                 f"_clip_{fmt(clip)}_layers_{n_layers}_units_{n_units}_seed{seed}.json")

                    p = _resolve_in_shards(dirpath, fname)
                    with open(p) as f:
                        data = json.load(f)
                    return _final_or_dict(data)

                elif search_space == "A2C":
                    dir_simple = os.path.join(self.data_path, "data_hpo_rl_bench", search_space, environment)
                    if set == "static":
                        fname = f"{environment}_{search_space}_random_lr_{lr}_gamma_{fmt(gamma)}_seed{seed}_eval.json"
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        fname = (f"{environment}_{search_space}_lr_{lr}_gamma_{fmt(gamma)}"
                                 f"_layers_{n_layers}_units_{n_units}_seed{seed}.json")

                    p = os.path.join(dir_simple, fname)
                    if not os.path.exists(p):
                        raise FileNotFoundError(p)
                    with open(p) as f:
                        data = json.load(f)
                    return _final_or_dict(data)

                elif search_space in ["DDPG", "TD3", "SAC"]:
                    tau = config.get("tau")
                    dirpath = os.path.join(self.data_path, "data_hpo_rl_bench", search_space, env_dir_for(search_space))

                    if set == "static":
                        # STATIC uses random_* and *_eval.json; no layers/units in filename
                        if search_space == "SAC":
                            n_layers = 2
                            n_units = 256
                            fname = (f"{environment}_{search_space}_lr_{lr}_gamma_{fmt(gamma)}_tau_{fmt(tau)}"
                                 f"_layers_{n_layers}_units_{n_units}_seed{seed}.json")
                        else:
                            fname = (f"{environment}_{search_space}_random_lr_{lr}"
                                     f"_gamma_{fmt(gamma)}_tau_{fmt(tau)}_seed{seed}_eval.json")
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        fname = (f"{environment}_{search_space}_lr_{lr}_gamma_{fmt(gamma)}_tau_{fmt(tau)}"
                                 f"_layers_{n_layers}_units_{n_units}_seed{seed}.json")

                    p = _resolve_in_shards(dirpath, fname)
                    with open(p) as f:
                        data = json.load(f)
                    return _final_or_dict(data)

            else:
                # multi-lr/gamma variant
                lrs = list(config.get("lr"))
                if len(lrs) == 1:
                    lrs = lrs * 3
                elif len(lrs) == 2:
                    lrs.append(lrs[1])
                gammas = list(config.get("gamma"))
                if len(gammas) == 1:
                    gammas = gammas * 3
                elif len(gammas) == 2:
                    gammas.append(gammas[1])

                if search_space in ["PPO", "TD3", "SAC"]:
                    dirpath = os.path.join(self.data_path, "data_hpo_rl_bench", search_space, env_dir_for(search_space))
                    fname = (f"{environment}_{search_space}_random_lr_{lrs[0]}{lrs[1]}{lrs[2]}"
                             f"_gamma_{gammas[0]}{gammas[1]}{gammas[2]}_seed{seed}_eval.json")
                    p = _resolve_in_shards(dirpath, fname)
                    with open(p) as f:
                        data = json.load(f)
                    return _final_or_dict(data)

        else:
            # online run mode
            for key in self.search_space_dict.keys():
                assert key in config.keys(), "The configuration must define a value for all the hyperparameters in the search space."
            data = run_rl_algorithm(
                rl_algorithm=self.rl_algorithm,
                rl_algorithm_name=self.search_space,
                config=config,
                environment=self.environment,
                seed=self.seed,
                total_timesteps=budget * 1e4,
            )
            return self._build_return_dict(data, budget)

    def get_search_space(self, search_space, set: str = "static"):
        if self.rl_algorithm is None:
            return self.search_space_structure[set][search_space]
        else:
            return self.search_space_dict

    def _precompute_configurations(self):
        self.precomputed_configurations = []
        search_space_structure = self.get_search_space(self.search_space, self.set)
        hps_names = list(search_space_structure.keys())
        for hps in itertools.product(*tuple(list(search_space_structure.values()))):
            self.precomputed_configurations.append(dict(zip(hps_names, hps)))

    def _coerce_hp_type(self, name, val):
        """Ensure consistent types for filename formatting & downstream code."""
        if name == "lr":
            return int(val)
        if name in {"gamma", "clip", "tau", "epsilon"}:
            return float(val)
        if name in {"n_layers", "n_units"}:
            return int(val)
        return val

    def sample_configuration(self):
        """
        Samples a random configuration from the current search space and set,
        coercing types so filenames are consistent.
        """
        configuration = {}
        space = self.get_search_space(self.search_space, self.set)  # dict
        for hp, values in space.items():
            choice = np.random.choice(values)
            configuration[hp] = self._coerce_hp_type(hp, choice.item() if hasattr(choice, "item") else choice)
        return configuration

    def run_bo(self, optimizer, iterations):
        observed_lc = {}
        configurations = self.precomputed_configurations.copy()

        next_conf_ix = np.random.randint(0, len(configurations))
        observed_lc[next_conf_ix] = self.get_metrics(configurations[next_conf_ix], budget=1)["eval_avg_returns"]

        for _ in range(iterations):
            next_conf_ix, budget = optimizer.observe_and_suggest(configurations, observed_lc)
            assert budget > 0, "Negative budgets are not allowed"
            assert budget < 100, "Upper bound of budget reached"

            observed_lc[next_conf_ix] = self.get_metrics(configurations[next_conf_ix], budget=budget)["eval_avg_returns"]

        max_per_lc = [max(lc) for lc in observed_lc.values()]
        return observed_lc, max_per_lc, configurations[np.argmax(max_per_lc)], list(observed_lc.keys())[np.argmax(max_per_lc)]

