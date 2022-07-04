#!/usr/bin/env python
import numpy as np
from math import exp
from math import sqrt

#ExecutionDynamics
class PoissonRate:
    def __init__(self, dt = 0.005, scale = 140.0, decay = 1.5):
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