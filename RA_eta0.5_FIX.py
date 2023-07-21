from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import numpy as np
from absl import logging
import tensorflow as tf
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.environments import tf_py_environment
from math import exp
from math import sqrt
from math import log
from tf_agents.environments import *
from matplotlib import scale
import time
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import numpy as np
from tf_agents.environments import *
from tf_agents.policies import *
import os
import functools
from absl import flags
from absl import app
import gin

from six.moves import range
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_dm_control
from tf_agents.environments import wrappers
from tf_agents.networks import sequential
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.keras_layers import dynamic_unroll_layer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.specs import tensor_spec
from tqdm import tqdm 

experiment_number='NQ1_'
adversary_name = experiment_number+'RA_eta0.5_adversary_FIX'
adversary_policyname = adversary_name+'_saved_policy'

MM_name = experiment_number+'RA_eta0.5_MM_FIX'
MM_policyname = MM_name+'_saved_policy'

MM_2actions_name = experiment_number+'RA_eta0.5_2action_FIX'
MM_2actions_policyname = MM_2actions_name+'_saved_policy'

MM_4actions_name = experiment_number+'RA_eta0.5_4action_FIX'
MM_4actions_policyname = MM_4actions_name+'_saved_policy'

tf.random.set_seed(42)


#ExecutionDynamics

class PoissonRate:
    def __init__(self, dt=0.005,scale = 140.0, decay = 1.5):
        self.dt = dt
        self.scale = scale
        self.decay = decay
    def match_prob(self, offset):
        lamb = self.scale * exp(-self.decay * offset)
        return min(1, max(0, lamb * self.dt))

#PriceDynamics

class BrownianMotion:
    def __init__(self, dt = 0.005, volatility = 2.0):
        self.dt = dt
        self.volatility = volatility
    
    def sample_increment(self, rng, x):
        w = rng.standard_normal()
        return self.volatility * sqrt(self.dt) * w

class BrownianMotionWithDrift:
    def __init__(self, dt = 0.005, drift = 0, volatility = 2.0):
        self.dt = dt
        self.volatility = volatility
        self.drift = drift
        
    def sample_increment(self, rng, x):
        w = rng.standard_normal()
        return self.drift * self.dt + self.volatility * sqrt(self.dt) * w

class OrnsteinUhlenbeck:
    def __init__(self, dt = 1.0, rate = 1.0, volatility = 1.0):
        self.dt = dt
        self.rate = rate
        self.volatility = volatility
        
    def sample_increment(self, rng, x):
        w = BrownianMotion(self.dt, self.volatility)
        return -self.rate * x * self.dt + w.sample_increment(rng, x)

class OrnsteinUhlenbeckWithDrift:
    def __init__(self, dt = 1.0, rate = 1.0, volatility = 1.0, drift = 0):
        self.dt = dt
        self.rate = rate
        self.volatility = volatility
        self.drift = drift
    
    def sample_increment(self, rng, x):
        w = BrownianMotion(self.dt, self.volatility)
        return -self.rate * (self.drift-x) * self.dt + w.sample_increment(rng, x)

class ASDynamics:
    def __init__(self, dt, price, rng, pd = None, ed = None):
        self._rng = rng
        self.dt = dt
        self.time = 0
        self.price = price
        self.price_initial = price
        self.price_dynamics = pd
        self.execution_dynamics = ed
        
    def set_price_dynamics(self, pd):
        self.price_dynamic = pd
    
    def set_execution_dynamics(self, ed):
        self.execution_dynamic = ed
    
    def innovate(self):
        rng = np.random.default_rng()
        price_inc = self.price_dynamics.sample_increment(rng, self.price)
        
        self.time += self.dt
        self.price += price_inc
        
        return price_inc
        
    def try_execute(self, offset):
        match_prob = self.execution_dynamics.match_prob(offset)
        #print("prob: "+str(match_prob))
        if self._rng.random() < match_prob:
            return offset
        
    def try_execute_ask(self, order_price):
        #print("try ask")
        offset = order_price - self.price
        return self.try_execute(offset)
        
    def try_execute_bid(self, order_price):
        #print("try bid")
        offset = self.price - order_price
        return self.try_execute(offset)

    def get_state(self):
        state = dict()
        state['dt'] = self.dt
        state['time'] = self.time
        state['price'] = self.price
        state['price_initial'] = self.price_initial
        return state


#Strategies

class LinearUtilityTerminalPenaltyStrategy:
    def __init__(self, k, eta):
        self.k = k
        self.eta = eta
    
    def compute(self, a, price, inventory):
        rp = price - 2.0 * inventory * self.eta
        sp = 2.0 / self.k + self.eta
        return (rp + sp / 2.0 - price, price - (rp - sp / 2.0))

class LinearUtilityStrategy:
    def __init__(self, k):
        self.k = k
    
    def compute(self, a, b, c):
        return (1.0 / self.k, 1.0 / self.k)

class ExponentialUtilityStrategy:
    def __init__(self, k, gamma, volatility):
        self.k = k
        self.gamma = gamma
        self.volatility = volatility
        
    def compute(self, time, price, inventory):
        gss = self.gamma * self.volatility * self.volatility
        rp = price - inventory * gss *(1.0 - time)
        sp = gss * (1.0 - time) + (2.0 / self.gamma) * log(1.0 + self.gamma / self.k)
        return (rp + sp / 2.0 - price, price - (rp - sp / 2.0))

def converter(x):
    return tf.convert_to_tensor([x])



#Define Env

INV_BOUNDS = (-50.0, 50.0)
DRIFT_BOUNDS = (0.0, 1.0)
SCALE_BOUNDS = (105.0, 175.0)
DECAY_BOUNDS = (1.125, 1.875)
WEALTH_BOUNDS = (-100000, 100000)
MAX_DRIFT = 5.0

#Adversary_RN_control_A

class AdversaryEnvironmentWithControllingA(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingA, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))
        
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((1,), np.float32, name='action', minimum=SCALE_BOUNDS[0], maximum=SCALE_BOUNDS[1])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        self.scale = action[0]
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        

        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Adversary_RN_control_Drift

class AdversaryEnvironmentWithControllingDrift(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingDrift, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))
        
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((1,), np.float32, name='action', minimum=DRIFT_BOUNDS[0], maximum=DRIFT_BOUNDS[1])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        drift = action[0]
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        
        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Adversary_RN_control_K

class AdversaryEnvironmentWithControllingK(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingK, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))
        
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((1,), np.float32, name='action', minimum=DECAY_BOUNDS[0], maximum=DECAY_BOUNDS[1])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        self.decay = action[0]
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        

        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Adversary_RN_control_All

class AdversaryEnvironmentWithControllingAll(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(AdversaryEnvironmentWithControllingAll, self).__init__(handle_auto_reset=True)
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))
        
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta=eta
        self.zeta = zeta
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def action_spec(self):
        return BoundedArraySpec((3,), np.float32, name = 'action', minimum = [DRIFT_BOUNDS[0], SCALE_BOUNDS[0], DECAY_BOUNDS[0]], maximum = [DRIFT_BOUNDS[1], SCALE_BOUNDS[1], DECAY_BOUNDS[1]])
    
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),
                                   PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        drift = action[0]
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.scale = action[1]
        self.decay = action[2]
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
        self.inv_strategy = LinearUtilityTerminalPenaltyStrategy(self.decay, self.eta)
        
        (ask_offset, bid_offset) = self.inv_strategy.compute(self.dynamics.time, self.dynamics.price, self.inv)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset

        self.reward = -(self.inv * self.dynamics.innovate()-self.zeta * self.inv**2)

        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward += self.eta * self.inv ** 2
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())

        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

    def _do_executions(self, ask_price, bid_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward -= ask_offset
                self.wealth += ask_price

        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward -= bid_offset
                self.wealth -= bid_price
    
    def is_terminal(self):
        return self.dynamics.time >= 1.0

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']



#MM_RN_control_A

class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingA(py_environment.PyEnvironment):

    def __init__(self,adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingA, self).__init__(
            handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]

        self.scale = float(adversary_action[0])
        
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']


#MM_RN_control_Drift

class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingDrift(py_environment.PyEnvironment):

    def __init__(self, adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingDrift, self).__init__(
            handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]
        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#MM_RN_control_K

class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingK(py_environment.PyEnvironment):

    def __init__(self, adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingK, self).__init__(
            handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]
        self.decay = float(adversary_action[0])
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#MM_RN_control_All

class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingAll(py_environment.PyEnvironment):

    def __init__(self, adversary_policy,eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingAll, self).__init__(
            handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = policy_step.action[0]


        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.scale = float(adversary_action[1])
        self.decay = float(adversary_action[2])

        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        self.adversary_policy_state = policy_step.state

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

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
        
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']




#MM_RN_fix

class MarketMakerEnvironmentAgainstFixedAdversary(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstFixedAdversary, self).__init__(
            handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#MM_RN_random


class MarketMakerEnvironmentAgainstRandomAdversary(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstRandomAdversary, self).__init__(handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        #self.dynamics = self.random_dynamics()

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None

    def random_dynamics(self):
        '''randomize dynamics
        drift bounds: [-5.0, 5.0]
        scale bounds: [105.0, 175.0]
        decay bounds: [1.125, 1.875]
        '''
        self.drift = - 5.0 + 10 * np.random.random()
        self.scale = 105 + 70 * np.random.random()
        self.decay = 1.125 + 0.75 * np.random.random()
        return ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt,self.drift, self.volatility), PoissonRate(self.dt, self.scale, self.decay))
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = self.random_dynamics()
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

class Evaluate_MarketMakerEnvironmentAgainstFixedAdversary(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(Evaluate_MarketMakerEnvironmentAgainstFixedAdversary, self).__init__(
            handle_auto_reset=True)

        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0

        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(seed=42), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.spread = None
        self.spread_list = []
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(seed=42), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.spread = None
        self.spread_list = []
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())

    def _step(self, action: np.ndarray):
        ask_offset = action[0]
        bid_offset = action[1]
        self.spread = ask_offset + bid_offset
        self.spread_list.append(self.spread)
        ask_price = self.dynamics.price + ask_offset
        bid_price = self.dynamics.price - bid_offset
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2

            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype=np.float32), self._discount,
                            self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype=np.float32), self._discount,
                        self._get_observation())

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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']
    
    def get_spread_list(self):
        return self.spread_list



actor_fc_layers=[64, 128, 256] 
actor_output_fc_layers=(64,) 
actor_lstm_size=(64,) 
actor_learning_rate=3e-4
critic_obs_fc_layers=[64, 128] 
critic_action_fc_layers=[64, 128] #new12 was[32, 64]
critic_joint_fc_layers=[64, 128, 256]
critic_output_fc_layers=(64,) 
critic_lstm_size=(64,) 
critic_learning_rate=3e-4
alpha_learning_rate=3e-4
target_update_tau=0.05
target_update_period=5 #1
td_errors_loss_fn=tf.math.squared_difference
gamma=0.99
reward_scale_factor=1 #0.1
gradient_clipping=None
debug_summaries=False
summarize_grads_and_vars=False
replay_buffer_capacity=100000
initial_collect_episodes=100
collect_episodes_per_iteration=1
use_tf_functions=True
summaries_flush_secs=10
num_eval_episodes=10 # better:200
observations_allowlist='position'
eval_metrics_callback=None
batch_size=64
train_sequence_length=20 #new12was15
num_iterations=100000
train_steps_per_iteration=200
eval_interval=200
log_interval=200
summary_interval=200
global_step = tf.compat.v1.train.get_or_create_global_step()

class Agent:
    def __init__(self,tf_env, eval_tf_env, name="agent"):

        self.tf_env = tf_env
        self.eval_tf_env = eval_tf_env
        self.time_step_spec = tf_env.time_step_spec()
        self.observation_spec = self.time_step_spec.observation
        self.action_spec = tf_env.action_spec()
        self.name = name

        
        self.rewards = []
        self.episodes = []
        
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            self.observation_spec,
            self.action_spec,
            input_fc_layer_params=actor_fc_layers,
            lstm_size=actor_lstm_size,
            output_fc_layer_params=actor_output_fc_layers,
            activation_fn=tf.keras.activations.relu,
            dtype=tf.float32,
            continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)

        critic_net = critic_rnn_network.CriticRnnNetwork(
            (self.observation_spec, self.action_spec),
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            lstm_size=critic_lstm_size,
            output_fc_layer_params=critic_output_fc_layers,
            kernel_initializer='glorot_uniform',
            last_kernel_initializer='glorot_uniform')

        self.tf_agent = sac_agent.SacAgent(
            self.time_step_spec,
            self.action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)
        
        self.tf_agent.initialize()
        
        self.eval_metrics = [
          tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
          tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
        ]
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]
        
        self.train_summary_writer = tf.compat.v2.summary.create_file_writer(
          name + "_train", flush_millis=summaries_flush_secs * 1000)
        self.train_summary_writer.set_as_default()

        self.eval_summary_writer = tf.compat.v2.summary.create_file_writer(
          name + "_eval", flush_millis=summaries_flush_secs * 1000)

        self.eval_policy = self.tf_agent.policy
        self.collect_policy = self.tf_agent.collect_policy
        
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)
        
        self.initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_episodes=initial_collect_episodes)

        self.collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.tf_env,
            self.collect_policy,
            observers=[self.replay_buffer.add_batch] + self.train_metrics,
            num_episodes=collect_episodes_per_iteration)        

        if use_tf_functions:
            self.initial_collect_driver.run = common.function(self.initial_collect_driver.run)
            self.collect_driver.run = common.function(self.collect_driver.run)
            self.tf_agent.train = common.function(self.tf_agent.train)
            
        logging.info(
            '----------------------- %s : Initializing replay buffer by collecting experience for %d episodes '
            'with a random policy.', self.name, initial_collect_episodes)
        self.initial_collect_driver.run()
        
        self.results = metric_utils.eager_compute(
            self.eval_metrics,
            self.eval_tf_env,
            self.eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=self.eval_summary_writer,
            summary_prefix='Metrics',
        )
        
        if eval_metrics_callback is not None:
            eval_metrics_callback(self.results, global_step.numpy())
        metric_utils.log_metrics(self.eval_metrics)

        
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=train_sequence_length + 1).prefetch(3)
        self.iterator = iter(self.dataset)
        
        if use_tf_functions:
            self.train_step = common.function(self.train_step)
        
    def train_step(self):
        experience, _ = next(self.iterator)
        return self.tf_agent.train(experience)
    
    def train(self):
        time_step = None
        policy_state = self.collect_policy.get_initial_state(self.tf_env.batch_size)

        saved_policy = tf.compat.v2.saved_model.load(MM_policyname)
        self.collect_policy = saved_policy

        timed_at_step = global_step.numpy()
        time_acc = 0
        for _ in tqdm(range(num_iterations)):
            start_time = time.time()
            time_step, policy_state = self.collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_loss = self.train_step()
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step.numpy(),
                             train_loss.loss)
                steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

            for train_metric in self.train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=self.train_metrics[:2])

            if global_step.numpy() % eval_interval == 0:
                self.results = metric_utils.eager_compute(
                    self.eval_metrics,
                    self.eval_tf_env,
                    self.eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=self.eval_summary_writer,
                    summary_prefix='Metrics',
                )
                self.rewards.append(float(self.results['AverageReturn']))
                self.episodes.append(_)
            if eval_metrics_callback is not None:
                eval_metrics_callback(self.results, global_step.numpy())
            metric_utils.log_metrics(self.eval_metrics) 
        
        self.save()
            
    def save(self):
        saver = PolicySaver(self.eval_policy, batch_size = None)
        saver.save(self.name+'_saved_policy')

def evaluation(policy, name,env,calculate_ratio, num_episodes=1000,num_times=10):

    terminal_wealth = []
    terminal_inv = []
    notquote_times =0
    askandbid_times =0
    onlyask_times =0
    onlybid_times =0
    act = [] #for 2actions:[1,0,1...];for 4actions:[1,0,2,3,4,1...]

    environment = tf_py_environment.TFPyEnvironment(env)

    for _ in tqdm(range(num_times)):

        for _ in range(num_episodes):

            time_step = environment.reset()
            policy_state = policy.get_initial_state(batch_size=1)
            
            while not time_step.is_last():
                policy_step = policy.action(time_step, policy_state)
                act.append(policy_step.action.numpy()[0])
                policy_state = policy_step.state
                time_step = environment.step(policy_step.action)
            terminal_inv.append(time_step.observation.numpy()[0][1])
            terminal_wealth.append(time_step.observation.numpy()[0][2])
            spread_list = environment.envs[0].get_spread_list()
            
    if calculate_ratio is True:
        for i in act: # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
            if i == 1: #quote(ask&bid)
                askandbid_times +=1
            elif i == 0: #not quote
                notquote_times += 1
            elif i ==2: #quote(ask)
                onlyask_times +=1
            elif i ==3: #quote(bid)
                onlybid_times +=1
            

    average_wealth=np.average(terminal_wealth)
    variance_wealth=np.std(terminal_wealth)
    if variance_wealth != 0:
        sharpe_ratio=average_wealth/variance_wealth
    else:
        sharpe_ratio='variance_wealth=0'
    min_wealth=min(terminal_wealth)
    max_wealth=max(terminal_wealth)

    average_inventory=np.average(terminal_inv)
    variance_inventory=np.std(terminal_inv)

    average_spread=np.average(spread_list)
    variance_spread=np.std(spread_list)

    if calculate_ratio is True:
        total_times=notquote_times+askandbid_times+onlyask_times+onlybid_times
        if total_times !=0:
            notquote_ratio=notquote_times/total_times
            askandbid_ratio=askandbid_times/total_times
            onlyask_ratio=onlyask_times/total_times
            onlybid_ratio=onlybid_times/total_times

            print('The following are the results of '+name+' evaluation:',
                '\naverage wealth:',average_wealth,
                '\nvariance wealth:',variance_wealth,
                '\nsharpe ratio:',sharpe_ratio,
                '\nmin wealth:',min_wealth,
                '\nmax wealth:',max_wealth,
                '\naverage inventory:',average_inventory,
                '\nvariance inventory:',variance_inventory,
                '\naverage spread:',average_spread,
                '\nvariance spread:',variance_spread,
                '\nnotquote_ratio: {:.2%}'.format(notquote_ratio),
                '\naskandbid_ratio: {:.2%}'.format(askandbid_ratio),
                '\nonlyask_ratio: {:.2%}'.format(onlyask_ratio),
                '\nonlybid_ratio: {:.2%}'.format(onlybid_ratio),flush=True)
        else:
            print('The following are the random policy results of ' +name+ ' :',
                '\naverage wealth:',average_wealth,
                '\nvariance wealth:',variance_wealth,
                '\nsharpe ratio:',sharpe_ratio,
                '\nmin wealth:',min_wealth,
                '\nmax wealth:',max_wealth,
                '\naverage inventory:',average_inventory,
                '\nvariance inventory:',variance_inventory,
                '\naverage spread:',average_spread,
                '\nvariance spread:',variance_spread,
                '\ntotal_times is 0',flush=True)
    else:
        print('The following are the results of '+name+' evaluation:',
            '\naverage wealth:',average_wealth,
            '\nvariance wealth:',variance_wealth,
            '\nsharpe ratio:',sharpe_ratio,
            '\nmin wealth:',min_wealth,
            '\nmax wealth:',max_wealth,
            '\naverage inventory:',average_inventory,
            '\nvariance inventory:',variance_inventory,
            '\naverage spread:',average_spread,
            '\nvariance spread:',variance_spread,flush=True)
  
    return terminal_wealth


def validate_with_random_policy(name,env,num_episodes=1000,num_times=10): 
    random_environment = tf_py_environment.TFPyEnvironment(env)
    random_policy = random_tf_policy.RandomTFPolicy(random_environment.time_step_spec(), random_environment.action_spec())
  
    random_wealth_list = []
    terminal_inv = []

    for _ in tqdm(range(num_times)):

        for _ in range(num_episodes):

            time_step = random_environment.reset()
            
            while not time_step.is_last():
                action_step = random_policy.action(time_step)
                time_step = random_environment.step(action_step.action)

            terminal_inv.append(time_step.observation.numpy()[0][1])
            random_wealth_list.append(time_step.observation.numpy()[0][2])

    average_wealth=np.average(random_wealth_list)
    variance_wealth=np.std(random_wealth_list)
    sharpe_ratio=average_wealth/variance_wealth
    min_wealth=min(random_wealth_list)
    max_wealth=max(random_wealth_list)

    average_inventory=np.average(terminal_inv)
    variance_inventory=np.std(terminal_inv)

    print('The following are the random policy results of ' +name+ ' :',
        '\naverage wealth:',average_wealth,
        '\nvariance wealth:',variance_wealth,
        '\nsharpe ratio:',sharpe_ratio,
        '\nmin wealth:',min_wealth,
        '\nmax wealth:',max_wealth,
        '\naverage inventory:',average_inventory,
        '\nvariance inventory:',variance_inventory,flush=True)

  
    return random_wealth_list


#train MM

num_iterations=100000
logging.set_verbosity(logging.INFO)
tf.compat.v1.enable_v2_behavior()

MM_env = MarketMakerEnvironmentAgainstFixedAdversary(eta=0.5, zeta=0.0)
MM_tf_env = tf_py_environment.TFPyEnvironment(MM_env)
MM_eval_env = tf_py_environment.TFPyEnvironment(MM_env)

MM_agent = Agent(MM_tf_env, MM_eval_env, name=MM_name)
MM_agent.train()

Evaluate_env_fixadversary = Evaluate_MarketMakerEnvironmentAgainstFixedAdversary(eta=0.5, zeta=0.0)

MM_saved_policy = tf.saved_model.load(MM_policyname)
MM_results=evaluation(policy=MM_saved_policy, name=MM_name,env=Evaluate_env_fixadversary,calculate_ratio=False)
MM_validate_results=validate_with_random_policy(name=MM_name,env=Evaluate_env_fixadversary)

# Define: 2actions_ControlA
class AllowNotQuoteEnvironment_ControlA(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlA, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state

            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state
        
        self.scale = float(adversary_action[0])
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
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
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: 2actions_ControlDrfit
class AllowNotQuoteEnvironment_ControlDrfit(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlDrfit, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state
        
        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
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
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: 2actions_ControlK
class AllowNotQuoteEnvironment_ControlK(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlK, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state

            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state
        

        self.decay = float(adversary_action[0])
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
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
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: 2actions_ControlAll
class AllowNotQuoteEnvironment_ControlAll(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlAll, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            
            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state

        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.scale = float(adversary_action[1])
        self.decay = float(adversary_action[2])

        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
       
        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
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
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: 2actions_FixedAdversary
class AllowNotQuoteEnvironment_Fix(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Fix, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)

        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state

            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

    
        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: 2actions_RandomAdversary
class AllowNotQuoteEnvironment_Random(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Random, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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

    def random_dynamics(self):
        '''randomize dynamics
        drift bounds: [-5.0, 5.0]
        scale bounds: [105.0, 175.0]
        decay bounds: [1.125, 1.875]
        '''
        self.drift = - 5.0 + 10 * np.random.random()
        self.scale = 105 + 70 * np.random.random()
        self.decay = 1.125 + 0.75 * np.random.random()
        return ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt,self.drift, self.volatility), PoissonRate(self.dt, self.scale, self.decay))
      
    def action_spec(self):
        # 0 -> not quote; 1 -> quote
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 1)
    
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = self.random_dynamics()
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)

        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state

            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
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
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: Evaluate: 2actions_fix
class Evaluate_AllowNotQuoteEnvironment_Fix(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(Evaluate_AllowNotQuoteEnvironment_Fix, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(seed=42), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.spread = None
        self.spread_list = []
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
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(seed=42), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.spread = None
        self.spread_list = []
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)

        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state

            ask_offset = float(quote_action[0])
            bid_offset = float(quote_action[1])
            self.spread = ask_offset + bid_offset
            self.spread_list.append(self.spread)  
            ask_price = self.dynamics.price + ask_offset
            bid_price = self.dynamics.price - bid_offset
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
            self._do_executions(ask_price, bid_price)

        else:
            self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2

    
        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
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

    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

    def get_spread_list(self):
        return self.spread_list

#Define: 4actions_ControlA
class AllowNotQuoteEnvironment_ControlA_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlA_4actions, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state


        self.scale = float(adversary_action[0])
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)


        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: 4actions_ControlDrfit
class AllowNotQuoteEnvironment_ControlDrfit_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlDrfit_4actions, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)       
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state

        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: 4actions_ControlK
class AllowNotQuoteEnvironment_ControlK_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlK_4actions, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state

        self.decay = float(adversary_action[0])
        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0

#Define: 4actions_ControlAll
class AllowNotQuoteEnvironment_ControlAll_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,adversary_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_ControlAll_4actions, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        self.adversary_policy = adversary_policy
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
    
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
  
        adversary_policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        adversary_action = adversary_policy_step.action[0]
        self.adversary_policy_state = adversary_policy_step.state
        

        drift = float(adversary_action[0])
        self.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.scale = float(adversary_action[1])
        self.decay = float(adversary_action[2])

        self.dynamics.execution_dynamics=PoissonRate(self.dt, self.scale, self.decay)
        self.dynamics.price_dynamics=BrownianMotionWithDrift(self.dt, self.drift, self.volatility)
    

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

# 4actions_FixedAdversary
class AllowNotQuoteEnvironment_Fix_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Fix_4actions, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#4actions_RandomAdversary
class AllowNotQuoteEnvironment_Random_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(AllowNotQuoteEnvironment_Random_4actions, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

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

    def random_dynamics(self):
        '''randomize dynamics
        drift bounds: [-5.0, 5.0]
        scale bounds: [105.0, 175.0]
        decay bounds: [1.125, 1.875]
        '''
        self.drift = - 5.0 + 10 * np.random.random()
        self.scale = 105 + 70 * np.random.random()
        self.decay = 1.125 + 0.75 * np.random.random()
        return ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(self.dt,self.drift, self.volatility), PoissonRate(self.dt, self.scale, self.decay))

    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = self.random_dynamics()
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

#Define: Evaluate: 4actions_fix
class Evaluate_AllowNotQuoteEnvironment_Fix_4actions(py_environment.PyEnvironment):
    def __init__(self, quote_policy,eta=0.0, zeta=0.0,discount = 1.0):
        super(Evaluate_AllowNotQuoteEnvironment_Fix_4actions, self).__init__(handle_auto_reset=True)
        
        self.dt=0.005
        self.volatility=2
        self.decay=1.5
        self.scale=140
        self.drift=0
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(seed=42), BrownianMotionWithDrift(self.dt, self.drift, self.volatility),PoissonRate(self.dt, self.scale, self.decay))

        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.spread = None
        self.spread_list = []
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.quote_policy = quote_policy
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
    def action_spec(self):
        # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
        return BoundedArraySpec((), np.int32, name = 'action', minimum = 0, maximum = 3)
       
    def observation_spec(self):
        return BoundedArraySpec((3,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0],WEALTH_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1], WEALTH_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv,self.wealth], dtype=np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(seed=42), BrownianMotionWithDrift(0.005, 0.0, 2.0),PoissonRate(0.005,140,1.5))
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.spread = None
        self.spread_list = []
        self.quote_policy_state = self.quote_policy.get_initial_state(batch_size=1)
  
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32), self._discount, self._get_observation())
    
    def _step(self, action: np.int32):
        quote = action
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        if quote:
            policy_step = self.quote_policy.action(time_step, self.quote_policy_state)
            quote_action = policy_step.action[0]
            self.quote_policy_state = policy_step.state
            if quote ==1: 
                ask_offset = float(quote_action[0])
                bid_offset = float(quote_action[1])
                self.spread = ask_offset + bid_offset
                self.spread_list.append(self.spread) 
                ask_price = self.dynamics.price + ask_offset
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
                self._do_executions_bid(bid_price)
            elif quote ==2:
                ask_offset = float(quote_action[0])
                ask_price = self.dynamics.price + ask_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_ask(ask_price)
            elif quote==3:
                bid_offset = float(quote_action[1])
                bid_price = self.dynamics.price - bid_offset
                self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2
                self._do_executions_bid(bid_price)
        else:
            self.reward = self.inv * self.dynamics.innovate()-self.zeta * self.inv**2

        if self.is_terminal():
            self.wealth += self.dynamics.price * self.inv
            self.reward -= self.eta * self.inv ** 2
            
            self.inv_terminal = self.inv
            return TimeStep(StepType.LAST, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
        return TimeStep(StepType.MID, np.asarray(self.reward, dtype = np.float32), self._discount, self._get_observation())
    
    def _do_executions_ask(self, ask_price):
        if self.inv > INV_BOUNDS[0]:
            ask_offset = self.dynamics.try_execute_ask(ask_price)
            if ask_offset:
                self.inv -= 1.0
                self.reward += ask_offset
                self.wealth += ask_price
    def _do_executions_bid(self, bid_price):
        if self.inv < INV_BOUNDS[1]:
            bid_offset = self.dynamics.try_execute_bid(bid_price)
            if bid_offset:
                self.inv += 1.0
                self.reward += bid_offset
                self.wealth -= bid_price
        
    def is_terminal(self):
        return self.dynamics.time >= 1.0
    
    def get_state(self):
        dynamics_state = self.dynamics.get_state()
        env_state = dict()
        env_state['_discount'] = copy.deepcopy(self._discount)
        env_state['inv'] = self.inv
        env_state['inv_terminal'] = self.inv_terminal
        env_state['reward'] = self.reward
        env_state['wealth'] = self.wealth
        return {**dynamics_state, **env_state}

    def set_state(self, state):
        self.dynamics.dt = state['dt']
        self.dynamics.time = state['time']
        self.dynamics.price = state['price']
        self.dynamics.price_initial = state['price_initial']
        self._discount = copy.deepcopy(state['_discount'])
        self.inv = state['inv']
        self.inv_terminal = state['inv_terminal']
        self.reward = state['reward']
        self.wealth = state['wealth']

    def get_spread_list(self):
        return self.spread_list

#Define: DQN agent
class QuoteAgent:
    def __init__(self,env,name):

        num_iterations=100000
        train_sequence_length=30
        # Params for QNetwork
        fc_layer_params=(64,128,256) 
        # Params for QRnnNetwork
        input_fc_layer_params=(64,128,256) 
        lstm_size=(64,128,256) 
        output_fc_layer_params=(2,) 

        # Params for collect
        initial_collect_steps=2000
        collect_steps_per_iteration=1
        epsilon_greedy=0.1 
        replay_buffer_capacity=100000
        # Params for target update
        target_update_tau=0.05
        target_update_period=5
        # Params for train
        train_steps_per_iteration=1
        batch_size=64
        learning_rate=3e-4
        n_step_update=1
        gamma=0.99
        reward_scale_factor=1.0
        gradient_clipping=None
        use_tf_functions=True
        # Params for eval
        num_eval_episodes=10 #better 200
        eval_interval=200 #1000
        # Params for checkpoints
        train_checkpoint_interval=10000
        policy_checkpoint_interval=5000
        rb_checkpoint_interval=20000
        # Params for summaries and logging
        log_interval=200 #1000
        summary_interval=200 #1000
        summaries_flush_secs=10
        debug_summaries=False
        summarize_grads_and_vars=False
        eval_metrics_callback=None
        KERAS_LSTM_FUSED = 2

        self.name=name
        self.env=env

        logits = functools.partial(
            tf.keras.layers.Dense,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2))

        dense = functools.partial(
            tf.keras.layers.Dense,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

        fused_lstm_cell = functools.partial(
            tf.keras.layers.LSTMCell, implementation=KERAS_LSTM_FUSED)


        def create_feedforward_network(fc_layer_units, num_actions):
            return sequential.Sequential(
            [dense(num_units) for num_units in fc_layer_units]
            + [logits(num_actions)])

        def create_recurrent_network(
            input_fc_layer_units,
            lstm_size,
            output_fc_layer_units,
            num_actions):
            rnn_cell = tf.keras.layers.StackedRNNCells(
                [fused_lstm_cell(s) for s in lstm_size])
            return sequential.Sequential(
                [dense(num_units) for num_units in input_fc_layer_units]
                + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
                + [dense(num_units) for num_units in output_fc_layer_units]
                + [logits(num_actions)])
        

        root_dir = os.path.expanduser(self.name)
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'eval')

        train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
        
        eval_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)] 
        
        global_step = tf.compat.v1.train.get_or_create_global_step()

        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

            tf_env = tf_py_environment.TFPyEnvironment(self.env)
            eval_tf_env = tf_py_environment.TFPyEnvironment(self.env)

            action_spec = tf_env.action_spec()
            num_actions = action_spec.maximum - action_spec.minimum + 1

            if train_sequence_length != 1 and n_step_update != 1:
                raise NotImplementedError(
                'train_eval does not currently support n-step updates with stateful '
                'networks (i.e., RNNs)')
            
            if train_sequence_length > 1:
                q_net = create_recurrent_network(
                input_fc_layer_params,
                lstm_size,
                output_fc_layer_params,
                num_actions)
            else:
                q_net = create_feedforward_network(fc_layer_params, num_actions)
                train_sequence_length = n_step_update
            
            tf_agent = dqn_agent.DqnAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                q_network=q_net,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=common.element_wise_squared_loss,
                gamma=gamma,
                train_step_counter=global_step,
                epsilon_greedy=epsilon_greedy,

                n_step_update=n_step_update,
                reward_scale_factor=reward_scale_factor,
                gradient_clipping=gradient_clipping,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars)
            
            tf_agent.initialize()

            replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf_agent.collect_data_spec,
                batch_size=tf_env.batch_size,
                max_length=replay_buffer_capacity)

            replay_buffer_observer = replay_buffer.add_batch 

            train_metrics = [
                tf_metrics.NumberOfEpisodes(),
                tf_metrics.EnvironmentSteps(),
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric(),
            ]
            
            eval_policy = tf_agent.policy
            collect_policy = tf_agent.collect_policy

            collect_driver = dynamic_step_driver.DynamicStepDriver(
                tf_env,
                collect_policy,
                observers=[replay_buffer_observer] + train_metrics,
                num_steps=collect_steps_per_iteration) # collect x steps for each training iteration 

            train_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir,
                agent=tf_agent,
                global_step=global_step,
                metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
            policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'policy'),
                policy=eval_policy,
                global_step=global_step)
            rb_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
                max_to_keep=1,
                replay_buffer=replay_buffer)
            
            train_checkpointer.initialize_or_restore()
            rb_checkpointer.initialize_or_restore()
            
            # Dataset generates trajectories with shape [Bx2x...]
            dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,
                num_steps=train_sequence_length + 1).prefetch(3)
        

            if use_tf_functions:
                # To speed up collect use common.function.
                collect_driver.run = common.function(collect_driver.run)
                tf_agent.train = common.function(tf_agent.train)
            
            logging.info(
                '----------------------- %s : Initializing replay buffer by collecting experience for %d steps with '
                'a random policy.', self.name, initial_collect_steps)

            initial_collect_policy = random_tf_policy.RandomTFPolicy(
                tf_env.time_step_spec(), tf_env.action_spec())
            
            dynamic_step_driver.DynamicStepDriver(
                tf_env,
                initial_collect_policy,
                observers=[replay_buffer_observer] + train_metrics,
                num_steps=initial_collect_steps).run()    
            
            
            results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
            )
            
            if eval_metrics_callback is not None:
                eval_metrics_callback(results, global_step.numpy())
            metric_utils.log_metrics(eval_metrics)
            
            
            time_step = None
            policy_state = collect_policy.get_initial_state(tf_env.batch_size) #policy_state = ()
            iterator = iter(dataset)

            timed_at_step = global_step.numpy()
            time_acc = 0
            
            def train_step(iterator):
                experience, _ = next(iterator)
                return tf_agent.train(experience)
            
            if use_tf_functions:
                train_step = common.function(train_step)                

            for _ in range(num_iterations):
                start_time = time.time()
                time_step, policy_state = collect_driver.run(time_step=time_step,policy_state=policy_state)
                for _ in range(train_steps_per_iteration):
                    train_loss = train_step(iterator)
                time_acc += time.time() - start_time

                if global_step.numpy() % log_interval == 0:
                    logging.info('step = %d, loss = %f', global_step.numpy(),
                                train_loss.loss)
                    steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                    logging.info('%.3f steps/sec', steps_per_sec)
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                    timed_at_step = global_step.numpy()
                    time_acc = 0

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=train_metrics[:2])

                if global_step.numpy() % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % eval_interval == 0:
                    results = metric_utils.eager_compute(
                        eval_metrics,
                        eval_tf_env,
                        eval_policy,
                        num_episodes=num_eval_episodes,
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix='Metrics',
                    )
                    if eval_metrics_callback is not None:
                        eval_metrics_callback(results, global_step.numpy())
                    metric_utils.log_metrics(eval_metrics)

        saver = PolicySaver(eval_policy, batch_size = None)
        saver.save(self.name + '_saved_policy')




#train 2actions_MM
logging.set_verbosity(logging.INFO)
tf.compat.v1.enable_v2_behavior()
notquote_env = AllowNotQuoteEnvironment_Fix(quote_policy=MM_saved_policy,eta=0.5, zeta=0.0)
notquote=QuoteAgent(env=notquote_env,name=MM_2actions_name)
notquote_saved_policy = tf.saved_model.load(MM_2actions_policyname)

Evaluate_env_fixadversary = Evaluate_AllowNotQuoteEnvironment_Fix(quote_policy=MM_saved_policy,eta=0.5, zeta=0.0)
notquote_results=evaluation(policy=notquote_saved_policy, name=MM_2actions_name,env=Evaluate_env_fixadversary,calculate_ratio=True)

class QuoteAgent:
    def __init__(self,env,name):

        num_iterations=100000
        train_sequence_length=30
        # Params for QNetwork
        fc_layer_params=(64,128,256) 
        # Params for QRnnNetwork
        input_fc_layer_params=(64,128,256) 
        lstm_size=(64,128,256) 
        output_fc_layer_params=(4,) 

        # Params for collect
        initial_collect_steps=2000
        collect_steps_per_iteration=1
        epsilon_greedy=0.1 
        replay_buffer_capacity=100000
        # Params for target update
        target_update_tau=0.05
        target_update_period=5
        # Params for train
        train_steps_per_iteration=1
        batch_size=64
        learning_rate=3e-4
        n_step_update=1
        gamma=0.99
        reward_scale_factor=1.0
        gradient_clipping=None
        use_tf_functions=True
        # Params for eval
        num_eval_episodes=10 #better 200
        eval_interval=200 #1000
        # Params for checkpoints
        train_checkpoint_interval=10000
        policy_checkpoint_interval=5000
        rb_checkpoint_interval=20000
        # Params for summaries and logging
        log_interval=200 #1000
        summary_interval=200 #1000
        summaries_flush_secs=10
        debug_summaries=False
        summarize_grads_and_vars=False
        eval_metrics_callback=None
        KERAS_LSTM_FUSED = 2

        self.name=name
        self.env=env

        logits = functools.partial(
            tf.keras.layers.Dense,
            activation=None,
            kernel_initializer=tf.random_uniform_initializer(minval=-0.03, maxval=0.03),
            bias_initializer=tf.constant_initializer(-0.2))

        dense = functools.partial(
            tf.keras.layers.Dense,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.compat.v1.variance_scaling_initializer(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

        fused_lstm_cell = functools.partial(
            tf.keras.layers.LSTMCell, implementation=KERAS_LSTM_FUSED)


        def create_feedforward_network(fc_layer_units, num_actions):
            return sequential.Sequential(
            [dense(num_units) for num_units in fc_layer_units]
            + [logits(num_actions)])

        def create_recurrent_network(
            input_fc_layer_units,
            lstm_size,
            output_fc_layer_units,
            num_actions):
            rnn_cell = tf.keras.layers.StackedRNNCells(
                [fused_lstm_cell(s) for s in lstm_size])
            return sequential.Sequential(
                [dense(num_units) for num_units in input_fc_layer_units]
                + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
                + [dense(num_units) for num_units in output_fc_layer_units]
                + [logits(num_actions)])
        

        root_dir = os.path.expanduser(self.name)
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'eval')

        train_summary_writer = tf.compat.v2.summary.create_file_writer(train_dir, flush_millis=summaries_flush_secs * 1000)
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(eval_dir, flush_millis=summaries_flush_secs * 1000)
        
        eval_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)] 
        
        global_step = tf.compat.v1.train.get_or_create_global_step()

        with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

            tf_env = tf_py_environment.TFPyEnvironment(self.env)
            eval_tf_env = tf_py_environment.TFPyEnvironment(self.env)

            action_spec = tf_env.action_spec()
            num_actions = action_spec.maximum - action_spec.minimum + 1

            if train_sequence_length != 1 and n_step_update != 1:
                raise NotImplementedError(
                'train_eval does not currently support n-step updates with stateful '
                'networks (i.e., RNNs)')
            
            if train_sequence_length > 1:
                q_net = create_recurrent_network(
                input_fc_layer_params,
                lstm_size,
                output_fc_layer_params,
                num_actions)
            else:
                q_net = create_feedforward_network(fc_layer_params, num_actions)
                train_sequence_length = n_step_update
            
            tf_agent = dqn_agent.DqnAgent(
                tf_env.time_step_spec(),
                tf_env.action_spec(),
                q_network=q_net,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=common.element_wise_squared_loss,
                gamma=gamma,
                train_step_counter=global_step,
                epsilon_greedy=epsilon_greedy,

                n_step_update=n_step_update,
                reward_scale_factor=reward_scale_factor,
                gradient_clipping=gradient_clipping,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars)
            
            tf_agent.initialize()

            replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=tf_agent.collect_data_spec,
                batch_size=tf_env.batch_size,
                max_length=replay_buffer_capacity)

            replay_buffer_observer = replay_buffer.add_batch 

            train_metrics = [
                tf_metrics.NumberOfEpisodes(),
                tf_metrics.EnvironmentSteps(),
                tf_metrics.AverageReturnMetric(),
                tf_metrics.AverageEpisodeLengthMetric(),
            ]
            
            eval_policy = tf_agent.policy
            collect_policy = tf_agent.collect_policy

            collect_driver = dynamic_step_driver.DynamicStepDriver(
                tf_env,
                collect_policy,
                observers=[replay_buffer_observer] + train_metrics,
                num_steps=collect_steps_per_iteration) # collect x steps for each training iteration 

            train_checkpointer = common.Checkpointer(
                ckpt_dir=train_dir,
                agent=tf_agent,
                global_step=global_step,
                metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
            policy_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'policy'),
                policy=eval_policy,
                global_step=global_step)
            rb_checkpointer = common.Checkpointer(
                ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
                max_to_keep=1,
                replay_buffer=replay_buffer)
            
            train_checkpointer.initialize_or_restore()
            rb_checkpointer.initialize_or_restore()
            
            # Dataset generates trajectories with shape [Bx2x...]
            dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=batch_size,
                num_steps=train_sequence_length + 1).prefetch(3)
        

            if use_tf_functions:
                # To speed up collect use common.function.
                collect_driver.run = common.function(collect_driver.run)
                tf_agent.train = common.function(tf_agent.train)
            
            logging.info(
                '----------------------- %s : Initializing replay buffer by collecting experience for %d steps with '
                'a random policy.', self.name, initial_collect_steps)

            initial_collect_policy = random_tf_policy.RandomTFPolicy(
                tf_env.time_step_spec(), tf_env.action_spec())
            
            dynamic_step_driver.DynamicStepDriver(
                tf_env,
                initial_collect_policy,
                observers=[replay_buffer_observer] + train_metrics,
                num_steps=initial_collect_steps).run()    
            
            
            results = metric_utils.eager_compute(
                eval_metrics,
                eval_tf_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
            )
            
            if eval_metrics_callback is not None:
                eval_metrics_callback(results, global_step.numpy())
            metric_utils.log_metrics(eval_metrics)
            
            
            time_step = None
            policy_state = collect_policy.get_initial_state(tf_env.batch_size) #policy_state = ()
            iterator = iter(dataset)

            timed_at_step = global_step.numpy()
            time_acc = 0
            
            def train_step(iterator):
                experience, _ = next(iterator)
                return tf_agent.train(experience)
            
            if use_tf_functions:
                train_step = common.function(train_step)                

            for _ in range(num_iterations):
                start_time = time.time()
                time_step, policy_state = collect_driver.run(time_step=time_step,policy_state=policy_state)
                for _ in range(train_steps_per_iteration):
                    train_loss = train_step(iterator)
                time_acc += time.time() - start_time

                if global_step.numpy() % log_interval == 0:
                    logging.info('step = %d, loss = %f', global_step.numpy(),
                                train_loss.loss)
                    steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                    logging.info('%.3f steps/sec', steps_per_sec)
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                    timed_at_step = global_step.numpy()
                    time_acc = 0

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                        train_step=global_step, step_metrics=train_metrics[:2])

                if global_step.numpy() % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=global_step.numpy())

                if global_step.numpy() % eval_interval == 0:
                    results = metric_utils.eager_compute(
                        eval_metrics,
                        eval_tf_env,
                        eval_policy,
                        num_episodes=num_eval_episodes,
                        train_step=global_step,
                        summary_writer=eval_summary_writer,
                        summary_prefix='Metrics',
                    )
                    if eval_metrics_callback is not None:
                        eval_metrics_callback(results, global_step.numpy())
                    metric_utils.log_metrics(eval_metrics)

        saver = PolicySaver(eval_policy, batch_size = None)
        saver.save(self.name + '_saved_policy')


#train 4actions_MM
logging.set_verbosity(logging.INFO)
tf.compat.v1.enable_v2_behavior()
notquote_4actions_env=AllowNotQuoteEnvironment_Fix_4actions(quote_policy=MM_saved_policy,eta=0.5, zeta=0.0)
notquote_4actions=QuoteAgent(env=notquote_4actions_env,name=MM_4actions_name)
notquote_4actions_saved_policy = tf.saved_model.load(MM_4actions_policyname)

Evaluate_env_fixadversary = Evaluate_AllowNotQuoteEnvironment_Fix_4actions(quote_policy=MM_saved_policy,eta=0.5, zeta=0.0)
notquote_4actions_results=evaluation(policy=notquote_4actions_saved_policy, name=MM_4actions_name,env=Evaluate_env_fixadversary,calculate_ratio=True)