from typing import Any, Dict
import optuna


def sample_ppo_params_extended(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])
    tau = trial.suggest_categorical("clip", [0.1, 0.2, 0.3])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    n_units = trial.suggest_categorical("n_units", [32, 64, 128])
    hyperparams = {
        "gamma": gamma,
        "clip": tau,
        "lr": lr,
        "n_layers": n_layers,
        "n_units": n_units
    }

    return hyperparams

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TRPO hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])
    tau = trial.suggest_categorical("clip", [0.2, 0.3, 0.4])
    hyperparams = {
        "gamma": gamma,
        "clip": tau,
        "lr": lr,
    }

    return hyperparams


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])

    hyperparams = {
        "gamma": gamma,
        "lr": lr
    }

    return hyperparams

def sample_a2c_params_extended(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    n_units = trial.suggest_categorical("n_units", [32, 64, 128])

    hyperparams = {
        "gamma": gamma,
        "lr": lr,
        "n_layers": n_layers,
        "n_units": n_units
    }

    return hyperparams

def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])
    tau = trial.suggest_categorical("epsilon", [0.1, 0.2, 0.3])

    hyperparams = {
        "gamma": gamma,
        "lr": lr,
        "epsilon": tau
    }

    return hyperparams

def sample_dqn_params_extended(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])
    tau = trial.suggest_categorical("epsilon", [0.1, 0.2, 0.3])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    n_units = trial.suggest_categorical("n_units", [32, 64, 128])

    hyperparams = {
        "gamma": gamma,
        "lr": lr,
        "epsilon": tau,
        "n_layers": n_layers,
        "n_units": n_units
    }

    return hyperparams

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.0001])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    n_units = trial.suggest_categorical("n_units", [32, 64, 128])
    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "lr": lr,
        "n_layers": n_layers,
        "n_units": n_units
    }

    return hyperparams


def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.0001])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    n_units = trial.suggest_categorical("n_units", [32, 64, 128])
    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "lr": lr,
        "n_layers": n_layers,
        "n_units": n_units
    }

    return hyperparams


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
    lr = trial.suggest_categorical("lr", [-6, -5, -4, -3, -2, -1])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.0001])
    n_layers = trial.suggest_categorical("n_layers", [1, 2, 3])
    n_units = trial.suggest_categorical("n_units", [32, 64, 128])
    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "lr": lr,
        "n_layers": n_layers,
        "n_units": n_units
    }

    return hyperparams


HYPERPARAMS_SAMPLER = {
    "A2C": sample_a2c_params,
    "A2C_extended": sample_a2c_params_extended,
    "DQN": sample_dqn_params,
    "DQN_extended":sample_dqn_params_extended,
    "DDPG": sample_ddpg_params,
    "SAC": sample_sac_params,
    "PPO": sample_ppo_params,
    "PPO_extended": sample_ppo_params_extended,
    "TD3": sample_td3_params,
}