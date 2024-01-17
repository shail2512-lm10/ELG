import torch
from torch.optim import Adam as Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml
import time
import vrplib

from generate_data import generate_vrp_data, VRPDataset
from CVRPModel import CVRPModel
from CVRPEnv import CVRPEnv
from utils import rollout, check_feasible, Logger, seed_everything


def predict(model, instance, env, aug_factor, eval_type):
    # test
    model.eval()
    model.requires_grad_(False)
    avg_cost = 0.
    start = time.time()
    env.load_vrplib_problem(instance, aug_factor=aug_factor)
    reset_state, _, _ = env.reset()
    
    with torch.no_grad():
        model.pre_forward(reset_state)
        solutions, probs, rewards = rollout(model=model, env=env, eval_type=eval_type)

    # Return
    aug_reward = rewards.reshape(aug_factor, env.multi_width)
    # shape: (augmentation, batch, pomo)
    max_pomo_reward, _ = aug_reward.max(dim=1)  # get best results from pomo
    # shape: (augmentation, batch)
    no_aug_cost = -max_pomo_reward.float()  # negative sign to make positive value
    no_aug_cost_mean = no_aug_cost.mean()

    max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
    # shape: (batch,)
    aug_cost = -max_aug_pomo_reward.float()  # negative sign to make positive value
    aug_cost_mean = aug_cost.mean()

    avg_cost += -rewards.max(1)[0].mean()
    best_idx = rewards.max(1)[1]
    best_sols = torch.take_along_dim(solutions, best_idx[:, None, None].expand(solutions.shape), dim=1)
    # # check feasible
    check_feasible(best_sols[0:1], reset_state.node_demand[0:1])
    end = time.time()

    print("Aug cost: {:.2f}".format(aug_cost_mean))
    print("no aug Avg cost: {:.2f}, Wall-clock time: {:.2f}s".format(no_aug_cost_mean, float(end - start)))
    print(f"Best solution: {best_sols[0][0].tolist()}")
    
    return avg_cost


if __name__ == "__main__":
    with open('ELG/CVRP/config.yml', 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file.read(), Loader=yaml.FullLoader)

    # params
    device = "cuda:{}".format(config['cuda_device_num']) if config['use_cuda'] else 'cpu'
    multiple_width = config['params']['multiple_width']
    test_size = config['params']['test_size']
    test_batch_size = config['params']['test_batch_size']
    load_checkpoint = config['load_checkpoint']
    test_data = config['test_filename']
    model_params = config['model_params']
    aug_factor = config['params']['aug_factor']
    eval_type = config['params']['eval_type']
    seed = config['seed']

    seed_everything(seed=seed)

    # load checkpoint
    model = CVRPModel(**model_params)
    checkpoint = torch.load(load_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Initialize env
    env = CVRPEnv(multi_width=multiple_width, device=device)

    # Dataset
    #test_set = VRPDataset(test_data, num_samples=test_size, test=True)
    #test_loader = DataLoader(test_set, batch_size=test_batch_size)

    #instance
    instance = vrplib.read_instance(test_data)

    # test
    predict(model, instance, env, aug_factor=aug_factor, eval_type=eval_type)