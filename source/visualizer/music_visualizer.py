from environments.music_world import MusicWorld, CompositionState
from typing import List, Dict
from constants.note import Note, Symbol

from random import random, choice

import numpy as np

import time
import joblib
import os

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
      if os.path.isfile('models/%s.pkl' % model):
        print("Model '%s' loaded" % ('models/%s.pkl' % model))
        self.action_vals = joblib.load('models/%s.pkl' % model)
      else:
        print("Model '%s' does not exist, one will be created" % ('models/%s.pkl' % model))

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
      joblib.dump(self.action_vals, 'models/%s.pkl' % model)
    
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