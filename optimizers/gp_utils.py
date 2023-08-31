import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
import copy
import numpy as np
import os
import time
import gpytorch
import logging


def EI(incumbent, model, likelihood, support_x, support_y, query, budget, inc_x, config):
    q_tuple = [query[i] for i in range(len(query))]
    if inc_x == q_tuple:
        return -np.inf
    if q_tuple not in support_x:
        x_query = np.array(query)
        mu, stddev = predict(model, likelihood, support_x, support_y, x_query, config=config)
        mu = mu.reshape(-1, )
        stddev = stddev.reshape(-1, )
        with np.errstate(divide='warn'):
            imp = mu - incumbent
            Z = imp / stddev
            score = imp * norm.cdf(Z) + stddev * norm.pdf(Z)
            return score.item()
    return -np.inf

def propose_location(incumbent, X_sample, Y_sample, gpr, likelihood, budget, dim, config_space, inc_x, max_Y, conf,
                     hp_names):
    '''
    Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    scores = []
    for x in config_space:
        config = []
        for hp in hp_names:
            config.append(x[hp])
        if x not in X_sample:
            acq = EI(incumbent/max_Y, gpr, likelihood, support_x=X_sample, support_y=Y_sample, query=config, budget=budget, inc_x=inc_x, config=conf)
        else:
            acq = -np.inf
        scores.append(acq)
    best_x = np.amax(scores)
    idxs = []
    for i, score in enumerate(scores):
        if score == best_x:
            idxs.append(i)
    if len(idxs) > 1:
        min_x = np.random.choice(idxs)
    else:
        min_x = np.argmax(scores)
    return min_x

def find_incumbent(y, budget):
    max_budget = np.amax([len(rs) for rs in y.values()])

    if max_budget >= budget:
        incumbent = -np.inf
        inc_x = None
        inc_budget = budget
        for config, rs in y.items():
            if len(rs) >= budget:
                if rs[budget-1] > incumbent:
                    incumbent = rs[budget-1]
                    inc_x = config
    else:
        incumbent = -np.inf
        inc_x = None
        inc_budget = max_budget
        for config, rs in y.items():
            if rs[-1] > incumbent:
                incumbent = rs[-1]
                inc_x = config
                inc_budget = len(rs)
    return incumbent, inc_x, inc_budget

def get_model_likelihood_mll(train_size, in_features, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.ones(train_size, in_features).to(device)
    train_y = torch.ones(train_size).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, config=config,
                         dims=in_features)
    model = model.to(device)
    likelihood = likelihood.to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)
    return model, likelihood, mll

def train(model, likelihood, mll, support_x, support_y, optimizer, epochs=1000, verbose=False, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = prepare_data(support_x, support_y)
    X = inputs
    max_Y = max(labels)
    if max_Y == 0: max_Y = 1
    labels /= max_Y
    Y = labels
    inputs, labels = torch.tensor(inputs, dtype=torch.float).to(device), torch.tensor(labels, dtype=torch.float).to(
        device)
    labels = labels / max(labels)
    losses = [np.inf]
    best_loss = np.inf
    starttime = time.time()
    patience = 0
    max_patience = config["patience"]
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        model.set_train_data(inputs=inputs, targets=labels, strict=False)
        predictions = model(inputs)
        try:
            loss = -mll(predictions, model.train_targets)
            loss.backward()
            optimizer.step()
        except Exception as ada:
            logging.info(f"Exception {ada}")
            break

        if verbose:
            print("Iter {iter}/{epochs} - Loss: {loss:.5f}   noise: {noise:.5f}".format(
                iter=_ + 1, epochs=epochs, loss=loss.item(),
                noise=likelihood.noise.item()))
        losses.append(loss.detach().to("cpu").item())
        if best_loss > losses[-1]:
            best_loss = losses[-1]
        # if np.allclose(losses[-1], losses[-2], atol=self.config["loss_tol"]):
        #     patience += 1
        # else:
        #     patience = 0
        # if patience > max_patience:
        #     break
    logging.info(
        f"Current Iteration: {len(Y)} | Incumbent {max(Y)} | Duration {np.round(time.time() - starttime)} | Epochs {_} | Noise {likelihood.noise.item()}")
    return model, likelihood, mll, max_Y


def prepare_data(support_x, support_y):
    inputs = []
    labels = []
    for cfg in support_y.keys():
        lc = support_y[cfg]
        inputs.append(np.array(list(cfg)))
        labels.append(lc[-1])
    return np.array(inputs), np.array(labels)


def predict(model, likelihood, support_x, support_y, query_x, noise_fn=None, config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = prepare_data(support_x, support_y)
    X = inputs
    max_Y = max(labels)
    if max_Y == 0: max_Y = 1
    labels /= max(labels)
    Y = labels / max_Y
    inputs, labels = torch.tensor(inputs, dtype=torch.float).to(device), torch.tensor(labels, dtype=torch.float).to(
        device)
    card = len(Y)
    if noise_fn:
        Y = noise_fn(Y)
    model.eval()
    likelihood.eval()
    model.set_train_data(inputs=inputs, targets=labels, strict=False)

    with torch.no_grad():
        query_x = torch.tensor(query_x, dtype=torch.float).to(device)
        pred = likelihood(model(torch.reshape(query_x, [1, config["dim"]])))

    mu = pred.mean.detach().to("cpu").numpy().reshape(-1, )
    stddev = pred.stddev.detach().to("cpu").numpy().reshape(-1, )

    return mu, stddev


class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,config,dims ):
        super(ExactGPLayer, self).__init__(train_inputs=train_x, train_targets=train_y, likelihood=likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

        self.normalizer = torch.nn.BatchNorm1d(num_features=dims, affine=False)
        ## RBF kernel
        if(config["kernel"]=='rbf' or config["kernel"]=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims if config["ard"] else None))
        elif(config["kernel"]=='52'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=config["nu"],ard_num_dims=dims if config["ard"] else None))
        ## Spectral kernel
        else:
            raise ValueError("[ERROR] the kernel '" + str(config["kernel"]) + "' is not supported for regression, use 'rbf' or 'spectral'.")

    def forward(self, x):
        x = self.normalizer(x)
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

