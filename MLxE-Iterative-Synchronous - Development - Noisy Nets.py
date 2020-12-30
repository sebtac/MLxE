# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:00:00 2020

@author: sebtac
@contact: https://www.linkedin.com/in/sebastian-taciak-5893861/

the MLxE - Iterative-Synchronous Architecture
    - Architecture for parallel implementation of the RL Algorithms
        - Implemented algorithm: A3C 
    - Python 3/Multiprocessing/TensorFlow 2
    - As many Executors as here are CPU cores on the Machine
    - Each Executor gerates only one example per iteration
    - Combined Memorizer and Learner executer iteratively with Executors

Test:
    - Noisy Nets
    - Two fold slowdown in speed of training
    - but better convergence guarantees.
    
"""

import multiprocessing as mp
import time
import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

import tensorflow_addons as tfa 

import math

# Assign Parameters
args = {# SYSTEM Settings
        "executors_n": mp.cpu_count(), # How many Executors are running in parallel
        #"executors_n": 1, # How many Executors are running in parallel

        # GYM Settings
        "max_episodes": 1000, # How many Episodes do you want to run?
        "env_name": 'CartPole-v0', # the name of the GYM Environment to run
        
        # MODEL Definition
        "state_n": 5, # value of env.observation_space() + all Task Specific Adjustments
        "action_n": 2, # value of env.action_space()
        "common_layers_n": [128,256,128], # Number of Neurons in Common Layers of A3C Model
        "value_layers_n": [64,128,64], # Number of Neurons in Value Layers of A3C Model
        "policy_layers_n": [64,128,64], # Number of Neurons in Policy Layers of A3C Model
         
        # LEARNER Settings
        "batch_size": 128, # Number of examples per model update
        "model_alignment_frequency": 128, # frequency of synchronization of the Target Model with On-Line Model -- OPTIONAL
        
        # LEARNING RATE Decceleration
        "lr_alpha": 0.0001, # Inital LR
        "lr_alpha_power": 0.998, #0.998 bast so far # Controls the pace of LR depreciation
        "lr_alpha_limit": 0.000001, # Lower limit on the LR
        
        # EXECUTOR Settings
        "internal_step_counter_limit": 50000, #limit of steps per episode 
        "experience_batch_size": 1024, # how many steps to save into the memory buffer from each run
        
        # advarse STATE Probability
        "prob_advarse_state_initial": 0.2, # Probability of choosing advarse case when starting new episode
        "prob_advarse_state_type_multiplier": 0.0, #Adjustment to the Probability to control the ratio of issue types that are promoted through advarse initialization
        
        # REWARD Incentives
        "reward_negative": -10.0, # Override the GYM's default negative return
        
        }

# Factorized Gaussian Noise Layer
# Reference from https://github.com/Kaixhin/Rainbow/blob/master/model.py
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, std_init=0.5, activation = "relu"):
        super().__init__()
        self.units = units
        self.std_init = std_init
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(input_dim, units), dtype='float32'),
                                     trainable=True)

        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(input_dim, units), dtype='float32'),
                                        trainable=True)

        self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(units,), dtype='float32'),
                                     trainable=True)

        self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(units,), dtype='float32'),
                                        trainable=True)

    def call(self, inputs):
        # output = tf.tensordot(inputs, self.kernel, 1)
        # tf.nn.bias_add(output, self.bias)
        # return output
        #print("self.weights_eps",self.weights_eps)
        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out

# Implementation that lets print out the values of the internals.        
class NoisyDense2(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, std_init=0.5, activation = "relu"):
        super().__init__()
        self.units = units
        self.std_init = std_init
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(input_dim, units), dtype='float32'),
                                     trainable=True)

        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(input_dim, units), dtype='float32'),
                                        trainable=True)

        self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(units,), dtype='float32'),
                                     trainable=True)

        self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(units,), dtype='float32'),
                                        trainable=True)

    def call(self, inputs):
        # output = tf.tensordot(inputs, self.kernel, 1)
        # tf.nn.bias_add(output, self.bias)
        # return output
        #print("self.weights_eps",self.weights_eps)
        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        
        print("1",self.weight_mu[0])
        print("2",self.weight_sigma[0])
        print("3",self.bias_mu[0])
        print("4",self.bias_sigma[0]) 
        print("5",self.kernel[0]) 
        print("6",self.bias[0])
        
        self.matmul = tf.matmul(inputs, self.kernel) + self.bias
        print("7",self.matmul[0])
        
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out
        
def LMxE(worker,
         lock,
         collect_obs,
         collect_examples,
         episod_counter,
         step_counter,
         executor_model,
         internal_step_counter_best,
         executor_counter,
         args):
    
    # Set Parameters
    executors_n = args["executors_n"]
    max_episodes = args["max_episodes"]
    env_name = args["env_name"]
    state_n = args["state_n"]
    action_n = args["action_n"]
    common_layers_n = args["common_layers_n"]
    value_layers_n = args["value_layers_n"]
    policy_layers_n = args["policy_layers_n"]
    batch_size = args["batch_size"]
    lr_alpha = args["lr_alpha"]
    lr_alpha_power = args["lr_alpha_power"]
    lr_alpha_limit = args["lr_alpha_limit"]
    prob_advarse_state_initial = args["prob_advarse_state_initial"]
    prob_advarse_state_type_multiplier = args["prob_advarse_state_type_multiplier"]
    internal_step_counter_limit = args["internal_step_counter_limit"]
    experience_batch_size = args["experience_batch_size"]
    reward_negative = args["reward_negative"]
    model_alignment_frequency = args["model_alignment_frequency"]

    # Assign EXECUTOR to all workers
    if worker >= 0:
        
        time_start = time.time()
        print("Starting E:", worker)
        
        # Establish Environment
        env = gym.make(env_name).unwrapped #unwrapped to access the behind the scenes elements of the environment
        
        # Define A3C Model for Executors
        inputs_executor = tf.keras.Input(shape=(state_n,))        
        
        common_network_executor = NoisyDense(units = common_layers_n[0], input_dim=state_n, activation='relu')(inputs_executor)
        common_network_executor = NoisyDense(units = common_layers_n[1], input_dim=common_layers_n[0], activation='relu')(common_network_executor)
        common_network_executor = NoisyDense(units = common_layers_n[2], input_dim=common_layers_n[1], activation='relu')(common_network_executor)
 
        policy_network_executor = NoisyDense(units = policy_layers_n[0], input_dim=common_layers_n[2], activation='relu')(common_network_executor)
        policy_network_executor = NoisyDense(units = policy_layers_n[1], input_dim=policy_layers_n[0], activation='relu')(policy_network_executor)
        policy_network_executor = NoisyDense(units = policy_layers_n[2], input_dim=policy_layers_n[1], activation='relu')(policy_network_executor)
        
        value_network_executor = NoisyDense(units = value_layers_n[0], input_dim=common_layers_n[2], activation='relu')(common_network_executor)
        value_network_executor = NoisyDense(units = value_layers_n[1], input_dim=value_layers_n[0], activation='relu')(value_network_executor)
        value_network_executor = NoisyDense(units = value_layers_n[2], input_dim=value_layers_n[1], activation='relu')(value_network_executor)
        
        logits_executor = NoisyDense(units = action_n, input_dim=policy_layers_n[2])(policy_network_executor)        
        values_executor = NoisyDense(units = 1, input_dim=value_layers_n[2])(value_network_executor)
        
        model_executor = Model(inputs=inputs_executor, outputs=[values_executor, logits_executor])
        
        # Define A3C Models for Learners        
        if worker == 0: #
            
            # Define BASE Model - Target
            inputs_base = tf.keras.Input(shape=(state_n,))
            
            common_network_base = NoisyDense(units = common_layers_n[0], input_dim=state_n, activation='relu')(inputs_base)
            common_network_base = NoisyDense(units = common_layers_n[1], input_dim=common_layers_n[0], activation='relu')(common_network_base)
            common_network_base = NoisyDense(units = common_layers_n[2], input_dim=common_layers_n[1], activation='relu')(common_network_base)
     
            policy_network_base = NoisyDense(units = policy_layers_n[0], input_dim=common_layers_n[2], activation='relu')(common_network_base)
            policy_network_base = NoisyDense(units = policy_layers_n[1], input_dim=policy_layers_n[0], activation='relu')(policy_network_base)
            policy_network_base = NoisyDense(units = policy_layers_n[2], input_dim=policy_layers_n[1], activation='relu')(policy_network_base)
            
            value_network_base = NoisyDense(units = value_layers_n[0], input_dim=common_layers_n[2], activation='relu')(common_network_base)
            value_network_base = NoisyDense(units = value_layers_n[1], input_dim=value_layers_n[0], activation='relu')(value_network_base)
            value_network_base = NoisyDense(units = value_layers_n[2], input_dim=value_layers_n[1], activation='relu')(value_network_base)
            
            logits_base = NoisyDense(units = action_n, input_dim=policy_layers_n[2])(policy_network_base)        
            values_base = NoisyDense(units = 1, input_dim=value_layers_n[2])(value_network_base)

            model_base = Model(inputs=inputs_base, outputs=[values_base, logits_base])
            
            
            # Define MAIN Model - Trainable Model
            inputs_main = tf.keras.Input(shape=(state_n,))
            
            common_network_main = NoisyDense(units = common_layers_n[0], input_dim=state_n, activation='relu')(inputs_main)
            common_network_main = NoisyDense(units = common_layers_n[1], input_dim=common_layers_n[0], activation='relu')(common_network_main)
            common_network_main = NoisyDense(units = common_layers_n[2], input_dim=common_layers_n[1], activation='relu')(common_network_main)
     
            policy_network_main = NoisyDense(units = policy_layers_n[0], input_dim=common_layers_n[2], activation='relu')(common_network_main)
            policy_network_main = NoisyDense(units = policy_layers_n[1], input_dim=policy_layers_n[0], activation='relu')(policy_network_main)
            policy_network_main = NoisyDense(units = policy_layers_n[2], input_dim=policy_layers_n[1], activation='relu')(policy_network_main)
            
            value_network_main = NoisyDense(units = value_layers_n[0], input_dim=common_layers_n[2], activation='relu')(common_network_main)
            value_network_main = NoisyDense(units = value_layers_n[1], input_dim=value_layers_n[0], activation='relu')(value_network_main)
            value_network_main = NoisyDense(units = value_layers_n[2], input_dim=value_layers_n[1], activation='relu')(value_network_main)
            
            logits_main = NoisyDense(units = action_n, input_dim=policy_layers_n[2])(policy_network_main)        
            values_main = NoisyDense(units = 1, input_dim=value_layers_n[2])(value_network_main)
            
            model_main = Model(inputs=inputs_main, outputs=[values_main, logits_main])
            
            # Define Optimizer
            optimizer = tfa.optimizers.RectifiedAdam(lr_alpha)
            
            lock.acquire()
            executor_model.append(model_main.get_weights()) # the first call MUST be append to create the entry [0]
            print("Saved Model", worker, len(executor_model))
            lock.release()

            memory_buffer = np.full(state_n+4,0.0)
            counter_learninig = 0

        while episod_counter.value < max_episodes:
            
            # Load Model
            if len(executor_model) > 0:
                lock.acquire()
                model_weights = executor_model[0]
                lock.release()
                model_executor.set_weights(model_weights)
                                           
            # Collect Examples & Save them in the Central Observation Repository
            current_state = env.reset()
            
            # ENSURE EXPLORATION OF advarse STATES
            if episod_counter.value <= 1:
                prob_advarse_state = prob_advarse_state_initial
            else:
                prob_advarse_state = np.clip(prob_advarse_state_initial/math.log(episod_counter.value,5), 0.05, 0.2)
            
            prob_random_state = 1-prob_advarse_state*4
            
            # CartPole position_start:
            # 0: Close to the Left Edge
            # 1: Close to the Right Edge
            # 2: Normal, random start (env.restart())
            # 3: Leaning Heavilly to the Left
            # 4: Leaning Heavilly to the Right
            
            # Choose one of the 5 scenarios with probabilities defined in p=()
            pos_start = np.random.choice(5,p=(prob_advarse_state+prob_advarse_state_type_multiplier*prob_advarse_state,
                                              prob_advarse_state+prob_advarse_state_type_multiplier*prob_advarse_state,
                                              prob_random_state,
                                              prob_advarse_state-prob_advarse_state_type_multiplier*prob_advarse_state,
                                              prob_advarse_state-prob_advarse_state_type_multiplier*prob_advarse_state))
            
            if pos_start == 0 or pos_start == 5:
                current_state[0] = -1.5 # -2.4 MIN
            if pos_start == 1 or pos_start == 6: 
                current_state[0] = 1.5 # -2.4 MAX
            if pos_start == 3:
                current_state[2] = -0.150 #-0.0.20943951023931953 MIN
            if pos_start == 4:
                current_state[2] = 0.150 #0.0.20943951023931953 MAX
            
            env.state = current_state
            
            # Custom State Representation Adjustment to help agent learn to be closer to the center
            current_state = np.append(current_state,current_state[0]*current_state[0]) 

            observations = np.empty(state_n+3)
            done = False
            internal_step_counter = 0
            
            while done == False and internal_step_counter <= internal_step_counter_limit:
                
                values, logits = model_executor(tf.convert_to_tensor(np.array(np.expand_dims(current_state,axis=0)), dtype=tf.float32))
                stochastic_action_probabilities = tf.nn.softmax(logits)
                action = np.random.choice(action_n, p=stochastic_action_probabilities.numpy()[0])
                next_state, reward, done, info = env.step(action)
                next_state = np.append(next_state,next_state[0]*next_state[0])

                # Add desired-behaviour incentive to the reward function
                R_pos = 1*(1-np.abs(next_state[0])/2.4) # 2.4 max value ### !!! in documentation it says 4.8 but failes beyound 2.4
                R_ang = 1*(1-np.abs(next_state[2])/0.20943951023931953) ### !!! in documentation it says 0.418 max value
                reward = reward + R_pos + R_ang
                
                # Custom Fail Reward to speed up Learning of conseqences of being in advarse position
                if done == True: 
                    reward = reward_negative # ST Original -1
                        
                current_observation = np.append(current_state,(reward, done, action))

                observations = np.vstack((observations, current_observation))
                current_state = next_state
                internal_step_counter += 1
                
                if internal_step_counter == 1:
                    observations = observations[1:]
                
                if done == True or internal_step_counter == internal_step_counter_limit:
                    
                    observations = observations[-np.minimum(observations.shape[0], 256):]
                    
                    exp_len = observations.shape[0]
                    exp_indices = np.array(range(exp_len)) + 1
                    rewards = np.flip(observations[:,5])
                    discounted_rewards = np.empty(exp_len)
                    reward_sum = 0
                
                    if observations[-1,-2] == 0:
                        observations[-1,-2] = 2                        
                        gamma = np.full(exp_len, 0.99)
                    else:
                        #print("exp_indices", exp_indices)
                        gamma = np.clip(0.0379 * np.log(exp_indices-1) + 0.7983, 0.5, 0.99)
                    if observations[-1,-2] == 1:
                        gamma[0] = 1
                    
                    for step in range(exp_len):
                        reward_sum = rewards[step] + gamma[step]*reward_sum
                        discounted_rewards[step] = reward_sum

                    discounted_rewards = np.flip(discounted_rewards)
                    
                    observations = np.hstack((observations,np.expand_dims(discounted_rewards, axis = 1)))
                                                                
                    lock.acquire()
                    collect_obs.put(observations)
                    lock.release()
                    
                    observations = np.empty(state_n+3)

            # Update Counters to Track Progress
            lock.acquire()
            episod_counter.value += 1
            lock.release()

            lock.acquire()    
            executor_counter.value += 1
            lock.release()
                
            print("Ending Executor:", worker, "Episod", episod_counter.value, "Initial State", pos_start, "Steps:", internal_step_counter)
            
            if internal_step_counter > internal_step_counter_best.value:
                internal_step_counter_best.value = internal_step_counter
                print("############################## BEST EPISOD LENGTH:", internal_step_counter, "Executor:", worker)                

            if internal_step_counter_best.value >= 50000:
                print("\nREACHED GOAL of 50K Steps in", internal_step_counter, "steps; Learning Iterations (Not Available); in",time.time()-time_start, "seconds \n")
            
            # Assign MEMORIZER to worker 0
            if worker == 0:
      
                while executor_counter.value < executors_n: # Starting Condition -- Allow to generate first traing example
                    pass

                while collect_obs.qsize() > 0:

                    #lock.acquire()
                    exp_temp = collect_obs.get()
                    #lock.release()
                    
                    memory_buffer = np.vstack((memory_buffer,exp_temp))
                    memory_buffer = memory_buffer[-np.minimum(memory_buffer.shape[0],experience_batch_size*executors_n):,:]

                batch_size_min = np.minimum(batch_size,memory_buffer.shape[0])
                runs = memory_buffer.shape[0] // np.minimum(memory_buffer.shape[0],batch_size_min) + 1
                
                for i in range(runs):

                    sample_index = np.random.choice(memory_buffer.shape[0],np.minimum(memory_buffer.shape[0],batch_size_min),replace=False)
                    sample = memory_buffer[sample_index, :]
    
                    #lock.acquire()
                    collect_examples.put(sample)
                    #lock.release()
                                    
                # Assign LEARNER to worker 0
                
                # Adjust Monotonically Decreasing Learning Rate
                #lock.acquire()
                next_lr_alpha = lr_alpha*np.power(lr_alpha_power,episod_counter.value)
                #lock.release()
                if next_lr_alpha < lr_alpha_limit:
                    next_lr_alpha = lr_alpha_limit
                
                optimizer.learning_rate = next_lr_alpha
                
                # Initialize LEARNER
                while collect_examples.qsize() > 0: # and episod_counter.value < max_episodes:
                    
                    #lock.acquire()
                    example = collect_examples.get()
                    #lock.release()

                    with tf.GradientTape() as tape:
                        values, logits = model_base(tf.convert_to_tensor(example[:,:5], dtype=tf.float32))
                        advantage = tf.convert_to_tensor(np.expand_dims(example[:,-1],axis=1), dtype=tf.float32) - values
                        value_loss = advantage ** 2 # this is a term to be minimized in trainig 
                        policy = tf.nn.softmax(logits)
                        entropy = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits), [-1,1])    
                        policy_loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=list(example[:,-2].astype(int)), logits=logits), [-1,1])            
                        policy_loss *= tf.stop_gradient(advantage) # advantage will be exluded from computation of the gradient; thsi allows to treat the values as constants
                        policy_loss -= 0.01 * entropy # entropy adjustment for better exploration 
                        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
                                            
                    grads = tape.gradient(total_loss, model_base.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model_main.trainable_weights))
                    
                    counter_learninig += 1
                
                    model_base.set_weights(model_main.get_weights()) ### the THREADED IMPLEMENTATION IS SYNCHRONIZED AT EACH STEP!!!
            
                
                #lock.acquire()
                executor_model[0] = model_main.get_weights()
                #lock.release()
                    
                #if counter_learninig % model_alignment_frequency == 0:
                model_base.set_weights(model_main.get_weights())                                    
                                
                #lock.acquire()
                executor_counter.value = 0
                #lock.release()
                
                print("LEARINING ITERATION:", counter_learninig,"\n")
                
            if worker > 0:
                while executor_counter.value > 0:
                    pass
                    
        print("FINAL EPISODE -- Best Episod Length:", internal_step_counter_best.value)
        print("Ending L:", worker)
        print("Ending Worker:", worker)

if __name__ == "__main__":
    
    # Create Shared Variables
    lock = mp.Lock() # LOCK object to ensure only one Processes can access the locked object
    manager = mp.Manager() # MANGER object to create the shared variables
    collect_obs = manager.Queue() # Collector of Episods from EXECUTORS
    #memory_buffer = manager.Queue() # Central Memmory Buffer -- implemented as numpy array within memorizer
    collect_examples = manager.Queue() # Collector of Training Examples
    episod_counter = manager.Value('i',0) # Number of Finalized Episodes
    step_counter = manager.Value('i',0) # Number of Finalized Steps
    internal_step_counter_best = manager.Value('i',0) # Length of the longest episod
    executor_model = manager.list()
    #executor_counter = manager.Array("i", [0 for i in range(args["executors_n"])])
    executor_counter = manager.Value('i',0)
    
    # Set Parallel Processes
    env_processes = []
    
    cores = args["executors_n"]
    
    print("CORES", cores)
    
    time_start = time.time()
    
    for worker in range(cores):
        p = mp.Process(target=LMxE, args=(worker,
                                          lock,
                                          collect_obs,
                                          collect_examples,
                                          episod_counter,
                                          step_counter,
                                          executor_model,
                                          internal_step_counter_best,
                                          executor_counter,
                                          args, ))
        
        p.start()
        env_processes.append(p)
    
    for p in env_processes:
        p.join()
        
    for p in env_processes:
        p.close()

    time_end = time.time()
    time_total = time_end - time_start
    print("Execution Time:", time_total)
    
    # Closing Actions
