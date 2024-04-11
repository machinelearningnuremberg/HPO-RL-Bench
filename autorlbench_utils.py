import json
import numpy as np
import os

DATA_PATH=os.getcwd()

def get_metrics(search_space: str, environment: str, config: dict, seed: int=0, budget: int=100, static: bool=True):
    if static:
        lr = config.get("lr")
        gamma = config.get("gamma")
        if search_space == "DQN":
            epsilon = config.get("epsilon")
            with open(os.path.join(DATA_PATH, 'data_hpo_rl_bench', search_space, environment,
                                   '%s_%s_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json'%(environment, search_space,
                                                                                    lr, gamma, epsilon, seed))) as f:
                data = json.load(f)
                train_timesteps_index = data["timesteps_train"].index(data["timesteps_eval"][budget-1])
            return {
                    "eval_avg_returns": data["returns_eval"][:budget],
                    "eval_std_returns": data["std_returns_eval"][:budget],
                    "eval_timestamps": data["timestamps_eval"][:budget],
                    "eval_timesteps": data["timesteps_eval"][:budget],
                    "train_timesteps": data["timesteps_train"][:train_timesteps_index],
                    "train_timestamps": data["timestamps_train"][:train_timesteps_index],
                    "train_returns": data["returns_train"][:train_timesteps_index]
                   }
        elif search_space == "PPO":
            clip = config.get("clip")
            with open(os.path.join(DATA_PATH, 'data_hpo_rl_bench', search_space, environment,
                                   '%s_%s_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json'%(environment, search_space,
                                                                                    lr, gamma, clip, seed))) as f:
                data = json.load(f)
                for i in range(budget):
                    timestep_eval = data["timesteps_eval"][i]
                    if np.isnan(timestep_eval):
                        data["timesteps_eval"][i] = (i+1) * 10000
            return {
                    "eval_avg_returns": data["returns_eval"][:budget],
                    "eval_std_returns": data["std_returns_eval"][:budget],
                    "eval_timestamps": data["timestamps_eval"][:budget],
                    "eval_timesteps": data["timesteps_eval"],
                   }
        elif search_space == "A2C":
            with open(os.path.join(DATA_PATH, 'data_hpo_rl_bench', search_space, environment,
                                   '%s_%s_random_lr_%s_gamma_%s_seed%s_eval.json'%(environment, search_space,
                                                                                    lr, gamma, seed))) as f:
                data = json.load(f)
            return {
                    "eval_avg_returns": data["returns_eval"][:budget],
                    "eval_std_returns": data["std_returns_eval"][:budget],
                    "eval_timestamps": data["timestamps_eval"][:budget],
                    "eval_timesteps": data["timesteps_eval"][:budget],
                   }
        elif search_space == "DDPG":
            tau = config.get("tau")
            with open(os.path.join(DATA_PATH, 'data_hpo_rl_bench', search_space, environment,
                                   '%s_%s_random_lr_%s_gamma_%s_tau_%s_seed%s_eval.json'%(environment, search_space,
                                                                                    lr, gamma, tau, seed))) as f:
                data = json.load(f)
            return {
                    "eval_avg_returns": data["returns_eval"][:budget],
                    "eval_std_returns": data["std_returns_eval"][:budget],
                    "eval_timestamps": data["timestamps_eval"][:budget],
                    "eval_timesteps": data["timesteps_eval"][:budget],
                   }
    else:
        lrs = config.get("lr")
        if len(lrs) == 1:
            lrs = lrs * 3
        elif len(lrs) == 2:
            lrs.append(lrs[1])

        gammas = config.get("gamma")
        if len(gammas) == 1:
            gammas = gammas * 3
        elif len(gammas) == 2:
            gammas.append(gammas[1])
        if search_space == "PPO":
            with open(os.path.join(DATA_PATH, 'data_hpo_rl_bench', search_space, environment,
                                   '%s_%s_random_lr_%s%s%s_gamma_%s%s%s_seed%s_eval.json'%(environment, search_space,
                                                                                           lrs[0], lrs[1], lrs[2],
                                                                                           gammas[0], gammas[1],
                                                                                           gammas[2], seed))) as f:
                data = json.load(f)
            return {
                    "eval_avg_returns": data["returns_eval"][:budget],
                    "eval_std_returns": data["std_returns_eval"][:budget],
                    "eval_timestamps": data["timestamps_eval"][:budget],
                    "eval_timesteps": data["timesteps_eval"][:budget]
                   }