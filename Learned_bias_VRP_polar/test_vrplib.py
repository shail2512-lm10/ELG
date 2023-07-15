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
import vrplib

from generate_data import generate_vrp_data, VRPDataset
from Att_CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible, Logger


def test(instance_list, model, env, max_size=200):
    # test
    # model.eval()
    avg_cost = 0.
    start = time.time()
    
    env.load_vrplib_problem_batch(instance_list, max_size=max_size)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)
    solutions, probs, rewards = rollout(model=model, env=env, eval_type='greedy')
    unscaled_reward = env.compute_unscaled_reward(solutions)

    optimal = np.array([instance[1] for instance in instance_list])
    cost = -unscaled_reward.max(1)[0]
    best_idx = unscaled_reward.max(1)[1]
    best_sols = torch.take_along_dim(solutions, best_idx[:, None, None].expand(solutions.shape), dim=1)
    # check feasible
    # check_feasible(best_sols[0:1], reset_state.node_demand[0:1, 0:175])
    end = time.time()
    avg_cost = cost.mean()
    avg_gap = ((cost.cpu().numpy() - optimal) / optimal).mean()
    print("Avg cost: {:.2f}, Avg optimal gap: {:.2f}%, Wall-clock time: {:2f}s".format(avg_cost, 100 * avg_gap, float(end - start)))
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
    # model = None

    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)

    # Dataset
    VRPLib_Path = "../VRPLib/Vrp-Set-X/X/"
    files = os.listdir(VRPLib_Path)
    small_instance = []
    medium_instance = []
    large_instance = []
    for file in files:
        if ".sol" in file:
            name = file[:-4]
        
            instance_file = VRPLib_Path + '/' + name + '.vrp'
            solution_file = VRPLib_Path + '/' + name + '.sol'
            instance = vrplib.read_instance(instance_file)
            solution = vrplib.read_solution(solution_file)

            problem_size = instance['node_coord'].shape[0] - 1
            if problem_size <= 200:
                small_instance.append([instance, solution['cost']])
            elif problem_size <= 500:
                medium_instance.append([instance, solution['cost']])
            elif problem_size <= 1000:
                large_instance.append([instance, solution['cost']])

    # test
    test(small_instance, model, env, 200)
    # test(medium_instance, model, env, 500)
    # test(large_instance, model, env, 1000)