import vrplib
import numpy as np
import torch
import yaml
import json
import wandb
import datetime
import time
import os
from torch.optim import Adam as Optimizer

from Att_CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible


def test_on_VRPLib_instance(config, model, name, repeat_idx=666, result_dict=None, instance=None, solution=None):
    torch.manual_seed(repeat_idx)
    # params
    # multiple_width = config['params']['multiple_width']

    # Read VRPLIB formatted instances
    instance = vrplib.read_instance(instance)
    solution = vrplib.read_solution(solution)
    optimal = solution['cost']
    problem_size = instance['node_coord'].shape[0] - 1
    multiple_width = min(problem_size, 1000)

    # Initialize CVRP state
    env = CVRPEnv(multiple_width, device)
    env.load_vrplib_problem(instance)

    reset_state, reward, done = env.reset()
    model.eval()
    model.requires_grad_(False)
    model.pre_forward(reset_state)

    # rollout
    actions, probs, reward = rollout(model, env, 'greedy')

    check_feasible(actions, env.reset_state.node_demand)
    unscaled_reward = env.compute_unscaled_reward(actions)
    print("best cost: {:.4f}".format(-float(unscaled_reward.max(1)[0])))
    if result_dict is not None:
        result_dict['best_so_far'] = -float(unscaled_reward.max(1)[0])
        result_dict['gap'] = (result_dict['best_so_far'] - optimal) / optimal

if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)
    experiment_type = 'final_run'
    # experiment_type = 'tune'
    repeat_times = 1
    instance_name = "Leuven1"
    if config['vrplib_set'] == 'X':
        VRPLib_Path = "../VRPLib/Vrp-Set-X/X/"
    # VRPLib_Path = "../VRPLib/A/"
    else:
        VRPLib_Path = "../VRPLib/Vrp-Set-XXL/Vrp-Set-XXL/XXL/selected_set"

    # params
    device = config['device']
    load_checkpoint = config['load_checkpoint']
    model_params = config['model_params']

    # Load RL policy
    model = CVRPModel(**model_params)
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize EAS layer
    # model.decoder.init_eas_layers_random(1)
    # model.decoder.enable_EAS = True

    model = model.to(device)

    if experiment_type == 'final_run':
        files = os.listdir(VRPLib_Path)
        vrplib_results = []
        for t in range(repeat_times):
            for name in files:
                if '.sol' in name:
                    continue
                name = name[:-4]
                instance_file = VRPLib_Path + '/' + name + '.vrp'
                solution_file = VRPLib_Path + '/' + name + '.sol'
                
                solution = vrplib.read_solution(solution_file)
                optimal = solution['cost']
                result_dict = {}
                result_dict['run_idx'] = t
                print(name)
                test_on_VRPLib_instance(config, model, repeat_idx=t, name=name, 
                                        result_dict=result_dict, instance=instance_file, solution=solution_file)

                # update the results of current instance and method
                exist = False
                for result_per_instance in vrplib_results:
                    if result_per_instance['instance'] == name:
                        exist = True  
                        result_per_instance['record'].append(result_dict)

                if exist == False:
                    new_instance_dict = {}
                    new_instance_dict['instance'] = name
                    new_instance_dict['optimal'] = optimal
                    new_instance_dict['record'] = [result_dict]
                    vrplib_results.append(new_instance_dict)

                print("Instance Name {}: gap {:.4f}".format(name, result_dict['gap']))

        with open(config['name'] + '_' + config['vrplib_set'] + '.json', 'w') as f:
            json.dump(vrplib_results, f)

        if config['vrplib_set'] == 'X':
            avg_gap_small = []
            avg_gap_medium = []
            avg_gap_large = []
            total = []
            for result in vrplib_results:
                scale = int(result['instance'].split('-')[1][1:])

                if scale <= 200:
                    avg_gap_small.append(result['record'][-1]['gap'])
                elif scale <= 500:
                    avg_gap_medium.append(result['record'][-1]['gap'])
                elif scale <= 1000:
                    avg_gap_large.append(result['record'][-1]['gap'])
                total.append(result['record'][-1]['gap'])
            
            print("{:.2f}%".format(100 * np.array(avg_gap_small).mean()))
            print("{:.2f}%".format(100 * np.array(avg_gap_medium).mean()))
            print("{:.2f}%".format(100 * np.array(avg_gap_large).mean()))
            print("{:.2f}%".format(100 *(np.array(total).mean())))

    
    elif experiment_type == 'tune':
        instance_file = VRPLib_Path + instance_name + '.vrp'
        solution_file = VRPLib_Path + instance_name + '.sol'
        test_on_VRPLib_instance(config, model, name=instance_file, 
                                instance=instance_file, solution=solution_file)
    
    elif experiment_type == 'repeat_run':
        instance_file = VRPLib_Path + instance_name + '.vrp'
        solution_file = VRPLib_Path + instance_name + '.sol'
        for t in range(repeat_times):
            test_on_VRPLib_instance(config, model, repeat_idx=t, 
                                    name=instance_file, instance=instance_file, solution=solution_file)
