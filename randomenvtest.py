import copy
import numpy as np

import tensorflow as tf
from tf_agents.environments import *
from tf_agents.specs import BoundedArraySpec, BoundedTensorSpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.policies import *
from dynamics import *
from strategies import *
from adversaryEnv import *
from marketMakerEnv import *
from allowNotQuoteEnv import *

def validate_with_random_policy(env):
    environment = tf_py_environment.TFPyEnvironment(env)
    rp = random_tf_policy.RandomTFPolicy(environment.time_step_spec(), environment.action_spec())
    policy = rp
    num_episodes = 200
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        all_return = []

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            all_return.append(episode_return)

    print('average:',np.average(all_return),'variance:',np.var(all_return))

#validate_with_random_policy()