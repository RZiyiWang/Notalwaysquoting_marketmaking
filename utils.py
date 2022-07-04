from tf_agents.environments import tf_py_environment
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def converter(x):
    return tf.convert_to_tensor([x])


def eval_policy(saved_policy, tf_env):
    total_return = 0.0
    policy = saved_policy
    num_episodes = 1000
    policy_state = saved_policy.get_initial_state(batch_size=1)
    mini = 200
    maxi = -200
    inter_min = 200
    reward = []
    for _ in range(num_episodes):

        time_step = tf_env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            policy_step = policy.action(time_step, policy_state)
            policy_state = policy_step.state
            time_step = tf_env.step(policy_step.action)
            episode_return += time_step.reward.numpy()[0]
            inter_min = min(inter_min, time_step.reward.numpy()[0])
        maxi = max(maxi, episode_return)
        mini = min(mini, episode_return)
        reward.append(episode_return)

    average_return = np.average(reward)
    variance_return = np.std(reward)
    print('average_return',average_return,'variance_return',variance_return)
    return reward

def plt_distribution(rewards):
    sns.kdeplot(rewards, shade=True)

