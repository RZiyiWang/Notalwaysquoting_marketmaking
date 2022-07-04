import copy
import numpy as np

import tensorflow as tf
from tf_agents.environments import *
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from dynamics import *
from strategies import *
from utils import *

INV_BOUNDS = (-50.0, 50.0)
MAX_DRIFT = 5.0

class AllowNotQuoteEnvironment(py_environment.PyEnvironment):
    def __init__(self, quote_policy, eta = 0.0, zeta =0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment, self).__init__(handle_auto_reset=True)
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotion(), PoissonRate())
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta
        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
    def observation_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'observation', minimum = [0.0, INV_BOUNDS[0]], maximum = [1.0, INV_BOUNDS[1]])
    
    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv], dtype = np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotion(), PoissonRate())
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        if quote:
            time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            self.quote_policy_state = policy_step.state
            quote_action = policy_step.action[0]
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
            
        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            #self.inv = 0.0
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
                
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0
