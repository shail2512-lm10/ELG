import torch
from torch.optim import Adam as Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import yaml
import wandb
import time
import os
import json
import tqdm

from generate_data import generate_vrp_data, VRPDataset
from Att_CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible, Logger


def test(dataloader, model, env):
    # test
    model.eval()
    avg_cost = 0.
    t = 0
    start = time.time()
    for batch in dataloader:
        env.load_random_problems(batch)
        reset_state, _, _ = env.reset()
        with torch.no_grad():
            model.pre_forward(reset_state)
            solutions, probs, rewards = rollout(model=model, env=env, eval_type='greedy')

        avg_cost += -rewards.max(1)[0].mean()
        best_idx = rewards.max(1)[1]
        best_sols = torch.take_along_dim(solutions, best_idx[:, None, None].expand(solutions.shape), dim=1)
        # check feasible
        check_feasible(best_sols[0:1], reset_state.node_demand[0:1])
        t += 1
    end = time.time()
    avg_cost /= t
    print("Avg cost: {}, Wall-clock time: {}s".format(avg_cost, float(end - start)))
    return avg_cost


if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # params
    device = config['device']
    multiple_width = config['params']['multiple_width']
    test_size = config['params']['test_size']
    test_batch_size = config['params']['test_batch_size']
    load_checkpoint = config['load_checkpoint']
    test_data = config['test_filename']
    model_params = config['model_params']

    # load checkpoint
    model = CVRPModel(**model_params)
    checkpoint = torch.load(load_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)

    # Dataset
    test_set = VRPDataset(test_data, num_samples=test_size, test=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size)

    # test
    test(test_loader, model, env)