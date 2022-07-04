import copy
from matplotlib import scale
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

class MarketMakerEnvironmentAgainstFixedAdversary(py_environment.PyEnvironment):

    def __init__(self, eta = 0.0, zeta=0.0, discount = 1.0):
        super(MarketMakerEnvironmentAgainstFixedAdversary, self).__init__(handle_auto_reset=True)
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta
        
        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'action', minimum = [0.0, 0.0], maximum = [3.0, 3.0])
    
    def observation_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'observation', minimum = [0.0, INV_BOUNDS[0]], maximum = [1.0, INV_BOUNDS[1]])
    
    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv], dtype = np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
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

class MarketMakerEnvironmentAgainstRandomAdversary(py_environment.PyEnvironment):

    def __init__(self, eta = 0.0, zeta=0.0, discount = 1.0):
        super(MarketMakerEnvironmentAgainstRandomAdversary, self).__init__(handle_auto_reset=True)
        self.dynamics = self.random_dynamics()
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta
        
        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        
    def random_dynamics(self):
        '''randomize dynamics
        drift bounds: [-5.0, 5.0]
        scale bounds: [105.0, 175.0]
        decay bounds: [1.125, 1.875]
        '''
        drift = - 5.0 + 10 * np.random.random()
        scale = 105 + 70 * np.random.random()
        decay = 1.125 + 0.75 * np.random.random()
        return ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(drift=drift), PoissonRate(scale=scale, decay=decay))
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'action', minimum = [0.0, 0.0], maximum = [3.0, 3.0])
    
    def observation_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'observation', minimum = [0.0, INV_BOUNDS[0]], maximum = [1.0, INV_BOUNDS[1]])
    
    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv], dtype = np.float32)
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

class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingDrift(py_environment.PyEnvironment):

    def __init__(self, eta = 0.0, zeta=0.0,discount = 1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingDrift, self).__init__(handle_auto_reset=True)
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta
        
        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.adversary_policy = tf.saved_model.load('adversaryWithControllingDrift_saved_policy')
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'action', minimum = [0.0, 0.0], maximum = [3.0, 3.0])
    
    def observation_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'observation', minimum = [0.0, INV_BOUNDS[0]], maximum = [1.0, INV_BOUNDS[1]])
    
    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv], dtype = np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
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
        
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = policy_step.state
        adversary_action = policy_step.action[0]
        drift = float(adversary_action[0])
        self.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)
        
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
    
class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingAll(py_environment.PyEnvironment):

    def __init__(self, eta = 0.0,zeta=0.0, discount = 1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingAll, self).__init__(handle_auto_reset=True)
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta
        
        self._discount = np.asarray(discount, dtype = np.float32)
        self._observation = None
        self.adversary_policy = tf.saved_model.load('adversaryWithControllingAll_saved_policy')
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)
        
    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'action', minimum = [0.0, 0.0], maximum = [3.0, 3.0])
    
    def observation_spec(self):
        return BoundedArraySpec((2,), np.float32, name = 'observation', minimum = [0.0, INV_BOUNDS[0]], maximum = [1.0, INV_BOUNDS[1]])
    
    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv], dtype = np.float32)
        return self._observation
    
    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
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
        
        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype = np.float32)), converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = policy_step.state
        adversary_action = policy_step.action[0]
        drift = float(adversary_action[0])
        scale = float(adversary_action[1])
        decay = float(adversary_action[2])
        self.dynamics.price_dynamics.drift = MAX_DRIFT * (2.0 * drift - 1.0)
        self.dynamics.execution_dynamics.scale = scale
        self.dynamics.execution_dynamics.decay = decay
        
        self.reward = self.inv * self.dynamics.innovate() -self.zeta * self.inv**2
        self._do_executions(ask_price, bid_price)
        
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



class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingA(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingA, self).__init__(
            handle_auto_reset=True)
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = tf.saved_model.load('adversaryWithControllingA_saved_policy')
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((2,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0]],maximum=[1.0, INV_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
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

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = policy_step.state
        adversary_action = policy_step.action[0]
        scale = float(adversary_action[0])
        self.dynamics.price_dynamics.scale = scale

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

class MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingK(py_environment.PyEnvironment):

    def __init__(self, eta=0.0, zeta=0.0, discount=1.0):
        super(MarketMakerEnvironmentAgainstStrategicAdversaryWithControllingK, self).__init__(
            handle_auto_reset=True)
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
        self.inv = 0.0
        self.inv_terminal = 0.0
        self.reward = 0.0
        self.wealth = 0.0
        self.eta = eta
        self.zeta = zeta

        self._discount = np.asarray(discount, dtype=np.float32)
        self._observation = None
        self.adversary_policy = tf.saved_model.load('adversaryWithControllingK_saved_policy')
        self.adversary_policy_state = self.adversary_policy.get_initial_state(batch_size=1)

    def action_spec(self):
        return BoundedArraySpec((2,), np.float32, name='action', minimum=[0.0, 0.0], maximum=[3.0, 3.0])

    def observation_spec(self):
        return BoundedArraySpec((2,), np.float32, name='observation', minimum=[0.0, INV_BOUNDS[0]],
                                maximum=[1.0, INV_BOUNDS[1]])

    def _get_observation(self):
        self._observation = np.asarray([self.dynamics.time, self.inv], dtype=np.float32)
        return self._observation

    def _reset(self):
        self.dynamics = ASDynamics(0.005, 100.0, np.random.default_rng(), BrownianMotionWithDrift(), PoissonRate())
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

        time_step = TimeStep(converter(StepType.MID), converter(np.asarray(self.reward, dtype=np.float32)),
                             converter(self._discount), converter(self._get_observation()))
        policy_step = self.adversary_policy.action(time_step, self.adversary_policy_state)
        self.adversary_policy_state = policy_step.state
        adversary_action = policy_step.action[0]
        decay = float(adversary_action[0])
        self.dynamics.price_dynamics.decay = decay

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