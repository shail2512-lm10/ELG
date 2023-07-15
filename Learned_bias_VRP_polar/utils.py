import torch
import numpy as np
import pandas as pd
import json


def rollout(model, env, eval_type='greedy'):
    env.reset()
    actions = []
    probs = []
    reward = None
    state, reward, done = env.pre_step()
    ins_feature = env.get_instance_feature()
    t = 0
    while not done:
        cur_dist, cur_theta = env.get_cur_feature()
        selected, one_step_prob = model.one_step_rollout(state, cur_dist, cur_theta, ins_feature, eval_type=eval_type)
        state, reward, done = env.step(selected)

        actions.append(selected)
        probs.append(one_step_prob)
        t += 1

    actions = torch.stack(actions, 1)
    if eval_type == 'greedy':
        probs = None
    else:
        probs = torch.stack(probs, 1)

    return torch.transpose(actions, 1, 2), probs, reward

def batched_two_opt_torch(cuda_points, cuda_tour, max_iterations=1000, device="cpu"):
  cuda_tour = torch.cat((cuda_tour, cuda_tour[:, 0:1]), dim=-1)
  iterator = 0
  problem_size = cuda_points.shape[0]
  with torch.inference_mode():
    batch_size = cuda_tour.shape[0]
    min_change = -1.0
    while min_change < 0.0:
      points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
      points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
      points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

      A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
      A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
      A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
      A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

      change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
      valid_change = torch.triu(change, diagonal=2)

      min_change = torch.min(valid_change)
      flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
      min_i = torch.div(flatten_argmin_index, problem_size, rounding_mode='floor')
      min_j = torch.remainder(flatten_argmin_index, problem_size)

      if min_change < -1e-6:
        for i in range(batch_size):
          cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
        iterator += 1
      else:
        break

      if iterator >= max_iterations:
        break

  return cuda_tour[:, :-1]

def check_feasible(pi, demand):
  # input shape: (1, multi, problem) 
  pi = pi.squeeze(0)
  multi = pi.shape[0]
  problem_size = demand.shape[1]
  demand = demand.expand(multi, problem_size)
  sorted_pi = pi.data.sort(1)[0]

  # Sorting it should give all zeros at front and then 1...n
  assert (
      torch.arange(1, problem_size + 1, out=pi.data.new()).view(1, -1).expand(multi, problem_size) ==
      sorted_pi[:, -problem_size:]
  ).all() and (sorted_pi[:, :-problem_size] == 0).all(), "Invalid tour"

  # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
  demand_with_depot = torch.cat(
      (
          torch.full_like(demand[:, :1], -1),
          demand
      ),
      1
  )
  d = demand_with_depot.gather(1, pi)

  used_cap = torch.zeros_like(demand[:, 0])
  for i in range(pi.size(1)):
      used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
      # Cannot use less than 0
      used_cap[used_cap < 0] = 0
      assert (used_cap <= 1 + 1e-4).all(), "Used more than capacity"


class Logger(object):
  def __init__(self, filename, config):
    '''
    filename: a json file
    '''
    self.filename = filename
    self.logger = config
    self.logger['result'] = {}
    self.logger['result']['val_100'] = []
    self.logger['result']['val_200'] = []
    self.logger['result']['val_500'] = []
    self.logger['result']['val_1000'] = []

  def log(self, info):
    '''
    Log validation cost on 4 datasets every 1000 steps
    '''
    self.logger['result']['val_100'].append(info[0].cpu().numpy().tolist())
    self.logger['result']['val_200'].append(info[1].cpu().numpy().tolist())
    self.logger['result']['val_500'].append(info[2].cpu().numpy().tolist())
    self.logger['result']['val_1000'].append(info[3].cpu().numpy().tolist())

    with open(self.filename, 'w') as f:
      json.dump(self.logger, f)