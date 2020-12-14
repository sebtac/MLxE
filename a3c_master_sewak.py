""" A3C in Code - Centralized/ Gobal Network Parameter Server/ Controller

Based On:
    
A3C Code as in the book Deep Reinforcement Learning, Chapter 12.

Runtime: Python 3.6.5
Dependencies: numpy, matplotlib, tensorflow (/ tensorflow-gpu), gym
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)
Inspired from: A3C implementation on TensorFLow official github repository (Tensorflow/models/research)

**********************************************************************

Adjusted by Seabstian Taciak as part of develeopment of MLxE Architecture

@author: sebtac
@contact: https://www.linkedin.com/in/sebastian-taciak-5893861/

"""

# SET BEFORE RUNNIG

# AGENT TYPE
# 0 - Sewak Base Agent (Fixed)
# 1 - Sewak DNN Adjusted
# 2 - Sewak "Task" Modified
# 3 - Sewak ISTB (Iterative, Synchronous Thread Based)

Agent_Type = 3

learning_rate = 0.0001

import multiprocessing
cores = multiprocessing.cpu_count() # DEFAULT SETTING
#cores = 1 # FOR DEBUGGING

# GENERAL IMPORTS
import sys
sys.path.append(r'C:\Users\surface\Documents\Python\RL\MLxE\Mohit Sewak RL\Mohit12_A3C')
import time
import winsound
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# DEEP LEARING and ENVIRONEMENT RELATER IMPORTS
import tensorflow as tf
import tensorflow_addons as tfa # ST for DNN Adjustment
import gym

# CUSTOM SEWAK's MODULES with OPTIONAL SEBTAC ADJUSTMENTS
from experience_replay_sewak import SimpleListBasedMemory

if Agent_Type == 0:
    from actorcritic_model_sewak import ActorCriticModel as ACModel # For Sewak Fixed version
    from a3c_worker_sewak_base import A3C_Worker # the intial Sewak's implementation with fixes of the Policy_Loss Calcultion
elif Agent_Type == 1:
    from actorcritic_model_sewak import ActorCriticModel_Dimond as ACModel
    from a3c_worker_sewak_DNN_Adjusted import A3C_Worker
elif Agent_Type == 2:
    from actorcritic_model_sewak import ActorCriticModel_Dimond as ACModel
    from a3c_worker_sewak_Task_Modifications import A3C_Worker
elif Agent_Type == 3:
    from actorcritic_model_sewak import ActorCriticModel_DoubleDimond as ACModel
    from a3c_worker_sewak_ISTB import A3C_Worker

# SEWAK's Implementation Fix
"""
- Policy Loss Calcualtion
- Using actual play in example generation (was random)
"""

# DNN Adjustments
"""
- Adding monotonic decrease in Learing Rate relative to the number of episodes run with:
    self.alpha_power = 0.998
    self.alpha_limit = 0.000001
- Modifying the Model to: common_network_size=[128,256,128], policy_network_size=[64,128,64], value_network_size=[64,128,64]
- Changing the Optimizer to RectifiedAdam -- requaires tensorflow_addons
- Changing Gamma coeffcient to 0.97
"""

# Task Specific Modifications
"""
- Modified state representation with addition of 5th parameter representing the squared distance of the cart from the center of the plane
- Adverse Initial Position
- Negative Reward: -10.0 (originally 0.0)
- Monotonically Decreasing Discount Factor (Gamma Coefficent)
- Goal Specific Reward for cart being close to center of the pland and the pole being close to vertical
"""

class A3C_Master():
    """A3C Master

        Centralized Master class of A3C used for hosting the global network parameters and spawning the agents.

        Args:
            env_name (str): Name of a valid gym environment
            model_dir (str): Directory for saving the model during training, and loading the same while playing
            learning_rate (float): The learning rate (alpha) for the optimizer

        Examples:
             agent = A3C_Master()
             agent.train()
             agent.play()

    """

    def __init__(self, Agent_Type=Agent_Type, env_name='CartPole-v0', model_dir="models", learning_rate=learning_rate): #ST 0.001 for Fixed, 0.0001 otherwise 
        self.env_name = env_name
        self.model_dir = model_dir
        self.alpha = learning_rate
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.env = gym.make(self.env_name)
        self.action_size = self.env.action_space.n
        
        if Agent_Type <= 1:
            self.state_size = self.env.observation_space.shape[0] # For None TaH imlementations
        elif Agent_Type == 2:
            self.state_size = self.env.observation_space.shape[0] + 1 # ST for TaH implementation
        elif Agent_Type == 3:
            self.state_size = self.env.observation_space.shape[0] + 1 # ST for TaH implementation
        
        if Agent_Type == 0:
            self.optimizer = tf.keras.optimizers.Adam(self.alpha)
        else:
            self.optimizer = tfa.optimizers.RectifiedAdam(self.alpha) # ST DNN Adjustment
        
        logger.debug("StateSize:{}, ActionSize:{}".format(self.state_size, self.action_size))
        self.master_model = ACModel(self.action_size)
        self.master_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self, cores):
        """Train the A3C agent
            Main function to train the A3C agent after instantiation.

            This method uses the number of processor cores to spawns as many Workers. The workers are spawned as
            multiple parallel threads instead of multiple parallel processes. Being a threaded execution, the workers
            share memory and hence can write directly into the shared global variables.

            A more optimal, completely asynchronous implementation could be to spawn the workers as different processes
            using a task queue or multiprocessing. In case if this is adopted, then the shared variables need to made
            accessible in the distributed environment.

        """

        a3c_workers = [A3C_Worker(self.master_model, 
                                  self.optimizer, 
                                  i, 
                                  self.env_name, 
                                  self.model_dir, 
                                  workers_num = cores, 
                                  learning_rate = learning_rate)
                       for i in range(cores)]
        for i, worker in enumerate(a3c_workers):
            logger.info("Starting worker {}".format(i))
            worker.start()
        [worker.join() for worker in a3c_workers]
        self.plot_training_statistics()

    def play(self):
        """Play the environment using a trained agent

            This function opens a (graphical) window that will play a trained agent. The function will try to retrieve
            the model saved in the model_dir with filename formatted to contain the associated env_name.
            If the model is not found, then the function will first call the train function to start the training.

        """
        env = self.env.unwrapped
        state = env.reset()
        model = self.master_model
        model_path = os.path.join(self.model_dir, 'model_{}.h5'.format(self.env_name))
        if not os.path.exists(model_path):
            logger.info('A3CMaster: No model found at {}, starting fresh training before playing!'.format(model_path))
            self.train()
        logger.info('A3CMaster: Playing env, Loading model from: {}'.format(model_path))
        print("Model Path:", model_path)
        #model.load_weights(model_path)
        done = False
        step_counter = 0
        reward_sum = 0
        try:
            while not done:
                env.render(mode='rgb_array')
                policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                logger.info("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()

    def plot_training_statistics(self, training_statistics=None):
        """Plot training statistics

        This function plot the training statistics like the steps, rewards, discounted_rewards, and loss in each
        of the training episode.

        """
        training_statistics = A3C_Worker.global_shared_training_stats if training_statistics is None \
            else training_statistics
        all_episodes = []
        all_steps = []
        all_rewards = []
        all_discounted_rewards = []
        all_losses = []
        for stats in training_statistics:
            worker, episode, steps, reward, discounted_rewards, loss = stats
            all_episodes.append(episode)
            all_steps.append(steps)
            all_rewards.append(reward)
            all_discounted_rewards.append(discounted_rewards)
            all_losses.append(loss)
        self._make_double_axis_plot(all_episodes, all_steps, all_rewards)
        self._make_double_axis_plot(all_episodes,all_discounted_rewards,all_losses, label_y1="Discounted Reward",
                                    label_y2="Loss", color_y1="cyan", color_y2="black")
        
        np.savetxt('run.csv', all_steps, delimiter=',', fmt='%d')

    @staticmethod
    def _make_double_axis_plot(data_x, data_y1, data_y2, x_label='Episodes (e)', label_y1='Steps To Episode Completion',
                               label_y2='Reward in each Episode', color_y1="red", color_y2="blue"):
        """Internal helper function for plotting dual axis plots
        """
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(label_y1, color=color_y1)
        ax1.plot(data_x, data_y1, color=color_y1)
        ax2 = ax1.twinx()
        ax2.set_ylabel(label_y2, color=color_y2)
        ax2.plot(data_x, data_y2, color=color_y2)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    """Main function for testing the A3C Master code's implementation
    """
    agent = A3C_Master(Agent_Type=Agent_Type)
    agent.train(cores)
    #agent.play()
    
    for i in range(10):
        winsound.Beep(500,500)
