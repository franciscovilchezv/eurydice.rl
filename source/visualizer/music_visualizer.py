from environments.music_world import MusicWorld, CompositionState
from typing import List, Dict
from constants.note import Note, Symbol

from models.music_world_nn import MusicWorldNN

from random import random, choice

import numpy as np

import time
import joblib
import os

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn

class InteractiveComposer:
  def __init__(self, env: MusicWorld):
    self.env: MusicWorld = env
    self.action_vals: Dict[CompositionState, Dict[int, float]] = dict() # Using Dict instead of List since total amount of states is unknown

    self.start_composition = [Symbol.NAN] * 8 # TODO: Think about a better representation of non-assigned values

    self.discount: float = 1 # TODO: Maybe add discount

    # self.states: List[CompositionState] = [] # Since we have too many states, we may not use this

  def greedy_policy_vis(self, num_steps: int):
    
    curr_state = CompositionState(self.start_composition)

    for itr in range(num_steps):
      action: Note = max(self.action_vals[curr_state], key=self.action_vals[curr_state].get)

      curr_state, _, _ = self.env.sample_transition(curr_state, action)

    print(curr_state)
    curr_state.play()
    time.sleep(1)

  def get_action_val(self, state: CompositionState, action: Note) -> float:
    if (state in self.action_vals and action in self.action_vals[state]):
      return self.action_vals[state][action]
    else:
      return 0.0 # TODO: define initial values for the state,action pairs

  def set_action_val(self, state: CompositionState, action: Note, val: float):
    if(not state in self.action_vals):
      self.action_vals[state] = dict()
    
    self.action_vals[state][action] = val

  def q_learning(self, epsilon: float, learning_rate: float, episodes: int, model: str, step: int):

    if(model):
      if os.path.isfile('trainings/%s.pkl' % model):
        print("Model '%s' loaded" % ('trainings/%s.pkl' % model))
        self.action_vals = joblib.load('trainings/%s.pkl' % model)
      else:
        print("Model '%s' does not exist, one will be created" % ('trainings/%s.pkl' % model))

    print("q-learning")
    state: CompositionState = CompositionState(self.start_composition)

    episode_num: int = 0

    print("Q-learning, episode %i" % episode_num)

    continuation = True
    while(continuation):
      if (self.env.is_terminal(state)):
        episode_num = episode_num + 1

        if (episode_num % step == 0):
          self.greedy_policy_vis(8)
        
        if (episode_num == episodes):
          break

        state = CompositionState(self.start_composition) # restart to initial state

        print("Q-learning, episode %i" % episode_num)

      state, continuation = self.q_learning_step(state, epsilon, learning_rate)
    
    if(model):
      joblib.dump(self.action_vals, 'trainings/%s.pkl' % model)
    
    print("DONE")


  def q_learning_step(self, state: CompositionState, epsilon: float, learning_rate: float):
    action: Note = self.get_random_action(state, epsilon)

    (state_next, reward, continuation) = self.env.sample_transition(state, action)

    if (not continuation):
      return state_next, continuation
    
    new_q_value = self.get_action_val(state, action) + learning_rate * (reward + ( self.discount * self.get_max_q_value_for_state(state_next) ) - self.get_action_val(state, action))
    self.set_action_val(state, action, new_q_value)

    return state_next, continuation

  def get_random_action(self, state: CompositionState, epsilon: float) -> Note:
    r : float = random()
    if (r < epsilon):
      return choice(self.env.get_actions())
    else:
      return self.get_best_action_for_state(state)

  def get_best_action_for_state(self, state: CompositionState):
    actions: List[Note] = self.env.get_actions()
    q_values: Dict[Note, float] = dict()

    for action in actions:
      q_values[action] = self.get_action_val(state, action)

    # Pick a random if multiple notes have max value
    max_value = q_values[max(q_values, key=q_values.get)]
    actions_with_max_value = []
    for action in actions:
      if(q_values[action] == max_value):
        actions_with_max_value.append(action)

    return choice(actions_with_max_value)

    # return max(q_values, key=q_values.get) # TODO: pick best note to follow if all equal

  def get_max_q_value_for_state(self, state: CompositionState):
    actions: List[Note] = self.env.get_actions()
    q_values: List[float] = []

    for action in actions:
      q_values.append(self.get_action_val(state, action))

    return max(q_values)

  def deep_q_learning(self, epsilon: float, learning_rate: float, batch_size: int, episodes: int, step: int):
    state: CompositionState = CompositionState(self.start_composition)

    torch.set_num_threads(1)
    device: torch.device = torch.device("cpu")
    dqn: nn.Module = MusicWorldNN()
    optimizer: Optimizer = optim.Adam(dqn.parameters(), lr=0.001)

    dqn_target: nn.Module = MusicWorldNN()
    dqn_target.eval()

    replay_buffer: List = []

    episode_num: int = 0
    update_num: int = 100
    total_steps: int = 0

    continuation = True

    print("Q-learning, episode %i" % episode_num)
    while continuation:
      dqn.eval()
      if self.env.is_terminal(state):
        episode_num = episode_num + 1

        if episode_num % step == 0:
          print("Visualizing greedy policy")
          self.greedy_policy_vis_dqn(40, dqn, device)
        
        if(episode_num == episodes):
          break

        state = CompositionState(self.start_composition)

        print("Q-learning, episode %i" % episode_num)

      state, dqn, replay_buffer = self.deep_q_learning_step(state, dqn, dqn_target,
                                  epsilon, self.discount, batch_size, optimizer, device, replay_buffer)

      if total_steps % update_num == 0:
        dqn_target.load_state_dict(dqn.state_dict())
        dqn_target.eval()

      if len(replay_buffer) > 10000:
        replay_buffer.pop(0)
      
      total_steps += 1
    
    print("DONE")


  def deep_q_learning_step(self, state: CompositionState, dqn: nn.Module, dqn_target: nn.Module, epsilon: float,
                           discount: float, batch_size: int, optimizer, device, replay_buffer: List):
    
    dqn.eval()

    # get action
    a = self.get_random_approximate_action(state, epsilon, dqn)

    # get transition
    (next_state, reward, _) = self.env.sample_transition(state,a)

    # add to replay buffer
    replay_buffer.append([state, a, reward, next_state])

    # sample from replay buffer and train
    batch_idxs = np.random.randint(len(replay_buffer), size=batch_size)

    states_nnet_np = np.concatenate([self.state_to_nnet_input(replay_buffer[idx][0]) for idx in batch_idxs], axis=0)
    actions_np = np.array([replay_buffer[idx][1] for idx in batch_idxs])
    rewards_np = np.array([replay_buffer[idx][2] for idx in batch_idxs])

    states_next = [replay_buffer[idx][3] for idx in batch_idxs]
    states_next_nnet_np = np.concatenate([self.state_to_nnet_input(replay_buffer[idx][3]) for idx in batch_idxs], axis=0)
    is_terminal_np = np.array([self.env.is_terminal(state_next) for state_next in states_next])

    states_nnet = torch.tensor(states_nnet_np, device=device)
    actions = torch.unsqueeze(torch.tensor(actions_np, device=device), 1)
    rewards = torch.tensor(rewards_np, device=device)
    states_next_nnet = torch.tensor(states_next_nnet_np, device=device)
    is_terminal = torch.tensor(is_terminal_np, device=device)

    # train DQN
    dqn.train()
    optimizer.zero_grad()      
    
    # compute target
    nnet_target_output = dqn_target(states_next_nnet.float())
    y_np = []

    for i in range(0,len(states_nnet)):
      r_i  = rewards[i]

      if(is_terminal[i]):
        y_i = r_i
      else:
        q_t_value = torch.max(nnet_target_output[i])
        y_i = r_i + (discount * q_t_value)

      y_np.append([y_i])

    y = torch.tensor(y_np).float()

    # get output of dqn
    nnet_output = dqn(states_nnet.float())
    nnet_output_indx = []
    
    nnet_output_np = []
    for i in range(0, len(nnet_output)):
      nnet_output_indx.append([actions[i]])

    nnet_outputs = nnet_output.gather(-1, torch.tensor(nnet_output_indx))

    # loss
    criterion = nn.MSELoss()
    loss = criterion(nnet_outputs, y)

    # backpropagation
    loss.backward()

    # optimizer step
    optimizer.step()

    return next_state, dqn, replay_buffer

  def get_random_approximate_action(self, state: CompositionState, epsilon: float, dqn: nn.Module) -> Note:
    r : float = random()
    tensor_nn_state = torch.tensor(self.state_to_nnet_input(state))
    actions_current_state = dqn(tensor_nn_state.float())

    if (r < epsilon):
      return torch.tensor(np.random.randint(len(actions_current_state[0])))
    else:
      return torch.argmax(actions_current_state)

  def state_to_nnet_input(self, state: CompositionState) -> np.ndarray:
    states_nnet = [int(n) + 1 for n in state.composition_notes]
    states_nnet = np.expand_dims(states_nnet, 0)

    return np.array(states_nnet)

  def greedy_policy_vis_dqn(self, num_steps: int, dqn: nn.Module, device):
    curr_state = CompositionState(self.start_composition)

    for itr in range(num_steps):
      if self.env.is_terminal(curr_state):
        break

      state_tens = torch.tensor(self.state_to_nnet_input(curr_state), device=device)
      action_vals_state = dqn(state_tens.float()).cpu().data.numpy()[0, :]

      action: Note = Note(np.argmax(action_vals_state))
      curr_state, _, _ = self.env.sample_transition(curr_state, action)

    print(curr_state)
    curr_state.play()
    time.sleep(1)

    print("")

