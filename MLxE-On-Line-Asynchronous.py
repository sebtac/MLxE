# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:00:00 2020

@author: sebtac
@contact: https://www.linkedin.com/in/sebastian-taciak-5893861/

the MLxE - On-Line-Asynchronous Architecture
    - Architecture for parallel implementation of the RL Algorithms
        - Implemented algorithm: A3C 
    - Python 3/Multiprocessing/TensorFlow 2
    - As many Executors as here are CPU cores on the Machine minus 2 for Memorizer and Learner Processes
    - Each Executor geretates examples continuouslu using the most recent model from the learner 
    - Memorizer in separate process 
    - Learner in separate process 
"""

import multiprocessing as mp
import time
import gym
import argparse
from copy import deepcopy
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

import tensorflow_addons as tfa

import math

# Assign Parameters
args = {# SYSTEM Settings
        "executors_n": mp.cpu_count() - 2, # How many Executors are running in parallel # -2 for Mamorizer and Learner

        # GYM Settings
        "max_episodes": 30000, # How many Episodes do you want to run?
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
        
        # ADVERSE STATE Probability
        "prob_adverse_state_initial": 0.2, # Probability of choosing adverse case when starting new episode
        "prob_adverse_state_type_multiplier": 0.0, #Adjustment to the Probability to control the ratio of issue types that are promoted through adverse initialization
        
        # REWARD Incentives
        "reward_negative": -10.0, # Override the GYM's default negative return
        
        }
    
def LMxE(worker,
         lock,
         collect_obs,
         collect_examples,
         episod_counter,
         step_counter,
         exemplar_model,
         internal_step_counter_best,
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
    prob_adverse_state_initial = args["prob_adverse_state_initial"]
    prob_adverse_state_type_multiplier = args["prob_adverse_state_type_multiplier"]
    internal_step_counter_limit = args["internal_step_counter_limit"]
    experience_batch_size = args["experience_batch_size"]
    reward_negative = args["reward_negative"]
    min_colexamp_size = 8
    min_colexp_size = 8
    model_alignment_frequency = args["model_alignment_frequency"]
    
    # Assign LEARNER to worker 0
    if worker == 0:
        print("Starting L:", worker)
        next_lr_alpha = 0 
        
        # Initialize LEARNER

        # Define A3C Model
        
        inputs_main = tf.keras.Input(shape=(state_n,))
        
        common_network_main = Dense(common_layers_n[0], activation='relu',name="1M")(inputs_main)
        common_network_main = Dense(common_layers_n[1], activation='relu',name="2M")(common_network_main)
        common_network_main = Dense(common_layers_n[2], activation='relu',name="3M")(common_network_main)

        policy_network_main = Dense(policy_layers_n[0], activation='relu',name="4M")(common_network_main)
        policy_network_main = Dense(policy_layers_n[1], activation='relu',name="5M")(policy_network_main)
        policy_network_main = Dense(policy_layers_n[2], activation='relu',name="6M")(policy_network_main)
        
        value_network_main = Dense(value_layers_n[0], activation='relu',name="7M")(common_network_main)
        value_network_main = Dense(value_layers_n[1], activation='relu',name="8M")(value_network_main)
        value_network_main = Dense(value_layers_n[2], activation='relu',name="9M")(value_network_main)

        logits_main = Dense(action_n,name="10M")(policy_network_main)
        values_main = Dense(1,name="11M")(value_network_main)

        model_main = Model(inputs=inputs_main, outputs=[values_main, logits_main])
        
        
        inputs_base = tf.keras.Input(shape=(state_n,))
        
        common_network_base = Dense(common_layers_n[0], activation='relu',name="1B")(inputs_base)
        common_network_base = Dense(common_layers_n[1], activation='relu',name="2B")(common_network_base)
        common_network_base = Dense(common_layers_n[2], activation='relu',name="3B")(common_network_base)

        policy_network_base = Dense(policy_layers_n[0], activation='relu',name="4B")(common_network_base)
        policy_network_base = Dense(policy_layers_n[1], activation='relu',name="5B")(policy_network_base)
        policy_network_base = Dense(policy_layers_n[2], activation='relu',name="6B")(policy_network_base)

        value_network_base = Dense(value_layers_n[0], activation='relu',name="7B")(common_network_base)
        value_network_base = Dense(value_layers_n[1], activation='relu',name="8B")(value_network_base)
        value_network_base = Dense(value_layers_n[2], activation='relu',name="9B")(value_network_base)
        
        logits_base = Dense(action_n,name="10B")(policy_network_base)
        values_base = Dense(1,name="11B")(value_network_base)            
        
        model_base = Model(inputs=inputs_base, outputs=[values_base, logits_base])
                
        model_base.set_weights(model_main.get_weights())
        
        optimizer = tfa.optimizers.RectifiedAdam(lr_alpha)
        
        lock.acquire()
        exemplar_model.append(model_main.get_weights()) # the first call MUST be append to create the entry [0]
        lock.release()
        
        counter_learninig = 0

        while episod_counter.value < max_episodes:
            
            while collect_examples.qsize() == 0:
                pass
            
            while collect_examples.qsize() > 0 and episod_counter.value <= max_episodes:
                
                lock.acquire()
                example = collect_examples.get()
                lock.release()
                
                lock.acquire()
                next_lr_alpha = lr_alpha*np.power(lr_alpha_power,episod_counter.value)
                lock.release()
                
                if next_lr_alpha < lr_alpha_limit:
                    next_lr_alpha = lr_alpha_limit
                
                with tf.GradientTape() as tape:
                    values, logits = model_base(tf.convert_to_tensor(example[:,:5], dtype=tf.float32))
                    advantage = tf.convert_to_tensor(np.expand_dims(example[:,-1],axis=1), dtype=tf.float32) - values
                    value_loss = advantage ** 2
                    policy = tf.nn.softmax(logits)
                    entropy = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits), [-1,1])    
                    policy_loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=example[:,-2].astype(int), logits=logits), [-1,1])            
                    # the next is the equation (Gradient(log(PI(A|S,THeta)*A))) that needs to be minimized
                    policy_loss *= tf.stop_gradient(advantage) # advantage will be exluded from computation of the gradient; thsi allows to treat the values as constants
                    policy_loss -= 0.01 * entropy # entropy adjustment for better exploration
                    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
                            
                grads = tape.gradient(total_loss, model_base.trainable_weights)
                optimizer.apply_gradients(zip(grads, model_main.trainable_weights))
                
                model_base.set_weights(model_main.get_weights()) ### the THREADED IMPLEMENTATION IS SYNCHRONIZED AT EACH STEP!!!
                
                lock.acquire()
                exemplar_model[0] = model_main.get_weights()
                lock.release()
                
                counter_learninig += 1
                
                print("Worker", worker, "LEARNING Iteration", counter_learninig, "Remaining Examples", collect_examples.qsize())
                           
        print("FINAL EPISODE -- Best Episod Length:", internal_step_counter_best.value)
        print("Ending L:", worker)
    
    # Assign MEMORIZER to worker 1
    if worker == 1:
        
        memory_buffer = np.full(state_n+4,0.0)
        
        print("Starting M:", worker)
        while episod_counter.value < max_episodes:
            while collect_obs.qsize() == 0: # Starting Condition -- Allow to generate first traing example
                pass
            
            if collect_obs.qsize() > 0:
                lock.acquire()
                exp_temp = collect_obs.get()
                lock.release()
                
                memory_buffer = np.vstack((memory_buffer,exp_temp))
                memory_buffer = memory_buffer[-np.minimum(memory_buffer.shape[0],experience_batch_size*executors_n):,:]                

                if collect_examples.qsize() == 0:
                    min_colexamp_size += 1
              
            # !!!!!!!!!! MAYBE NOT NECESSERY IN THIS VERSION!!!!!!!!!!!
            if episod_counter.value == 1:
                memory_buffer = memory_buffer[1:]
            
            internal_count = 0
            
            if memory_buffer.shape[0] >= batch_size:
                while internal_count <= min_colexamp_size:
    
                    sample_index = np.random.choice(memory_buffer.shape[0],np.minimum(memory_buffer.shape[0],batch_size),replace=False)
                    sample = memory_buffer[sample_index, :]
                    
                    lock.acquire()
                    if collect_examples.full():
                        collect_examples.get()
                    collect_examples.put(sample)
                    lock.release()
                    
                    internal_count += 1
    
        print("Ending M:", worker)
        
    # Assign EXECUTOR to the remaing workers
    if worker > 1:
        time_start = time.time()
        print("Starting E:", worker)
        # Establish Environment
        env = gym.make(env_name).unwrapped #unwrapped to access the behind the scenes elements of the environment
        
        # Define A3C Model
        inputs_executor = tf.keras.Input(shape=(state_n,))        
        
        common_network_executor = Dense(common_layers_n[0], activation='relu',name="1E")(inputs_executor)
        common_network_executor = Dense(common_layers_n[1], activation='relu',name="2E")(common_network_executor)
        common_network_executor = Dense(common_layers_n[2], activation='relu',name="3E")(common_network_executor)
 
        policy_network_executor = Dense(policy_layers_n[0], activation='relu',name="4E")(common_network_executor)
        policy_network_executor = Dense(policy_layers_n[1], activation='relu',name="5E")(policy_network_executor)
        policy_network_executor = Dense(policy_layers_n[2], activation='relu',name="6E")(policy_network_executor)
        
        value_network_executor = Dense(value_layers_n[0], activation='relu',name="7E")(common_network_executor)
        value_network_executor = Dense(value_layers_n[1], activation='relu',name="8E")(value_network_executor)
        value_network_executor = Dense(value_layers_n[2], activation='relu',name="9E")(value_network_executor)
        
        logits_executor = Dense(action_n,name="10E")(policy_network_executor)        
        values_executor = Dense(1,name="11E")(value_network_executor)
        
        model_executor = Model(inputs=inputs_executor, outputs=[values_executor, logits_executor])
            
        while episod_counter.value < max_episodes:
            
            # Load Model
                        
            while len(exemplar_model) == 0:
                pass
            
            if len(exemplar_model) > 0:
                lock.acquire()
                model_weights = exemplar_model[0]
                lock.release()
                
                model_executor.set_weights(model_weights)
                                          
            # Collect Examples & Save them in the Central Observation Repository
            current_state = env.reset()
            
            # ENSURE EXPLORATION OF ADVERSE STATES
            if episod_counter.value <= 1:
                prob_adverse_state = prob_adverse_state_initial
            else:
                prob_adverse_state = np.clip(prob_adverse_state_initial/math.log(episod_counter.value,5), 0.05, 0.2)
            
            prob_random_state = 1-prob_adverse_state*4
            
            # CartPole position_start:
            # 0: Close to the Left Edge
            # 1: Close to the Right Edge
            # 2: Normal, random start (env.restart())
            # 3: Leaning Heavilly to the Left
            # 4: Leaning Heavilly to the Right
            
            # Choose one of the 5 scenarios with probabilities defined in p=()
            pos_start = np.random.choice(5,p=(prob_adverse_state+prob_adverse_state_type_multiplier*prob_adverse_state,
                                              prob_adverse_state+prob_adverse_state_type_multiplier*prob_adverse_state,
                                              prob_random_state,
                                              prob_adverse_state-prob_adverse_state_type_multiplier*prob_adverse_state,
                                              prob_adverse_state-prob_adverse_state_type_multiplier*prob_adverse_state))
            
            if pos_start == 0 or pos_start == 5:
                current_state[0] = -1.5 # -2.4 MIN
            if pos_start == 1 or pos_start == 6: 
                current_state[0] = 1.5 # -2.4 MAX
            if pos_start == 3:
                current_state[2] = -0.150 #-0.0.20943951023931953 MIN
            if pos_start == 4:
                current_state[2] = 0.150 #0.0.20943951023931953 MAX
            
            env.state = current_state            
            current_state = np.append(current_state,current_state[0]*current_state[0]) # Custom State Representation Adjustment to help agent learn to be closer to the center
            observations = np.empty(state_n+3)
            done = False
            internal_step_counter = 0
            
            while not done and internal_step_counter <= internal_step_counter_limit:
                values, logits = model_executor(tf.convert_to_tensor(np.array(np.expand_dims(current_state,axis=0)), dtype=tf.float32))
                stochastic_action_probabilities = tf.nn.softmax(logits)
                action = np.random.choice(action_n, p=stochastic_action_probabilities.numpy()[0])
                next_state, reward, done, info = env.step(action)
                next_state = np.append(next_state,next_state[0]*next_state[0])
 
               # ST add desired-behaviour incentive to the reward function
                R_pos = 1*(1-np.abs(next_state[0])/2.4) # 4.8 max value
                R_ang = 1*(1-np.abs(next_state[2])/0.20943951023931953) # 0.418 max value
                reward = reward + R_pos + R_ang
                
                if done == True: 
                    reward = reward_negative # ST Original -1
                        
                current_observation = np.append(current_state,(reward, done, action))
                observations = np.vstack((observations, current_observation))
                current_state = next_state
                internal_step_counter += 1
                
                if internal_step_counter % experience_batch_size == 0 or done:
                    observations = observations[1:]
                    
                    exp_len = observations.shape[0]
                    exp_indices = np.array(range(exp_len)) + 1
                    rewards = np.flip(observations[:,5])
                    discounted_rewards = np.empty(exp_len)
                    reward_sum = 0
                
                    if observations[-1,-2] == 0:
                        observations[-1,-2] = 2                        
                        gamma = np.full(exp_len, 0.99)
                    else:
                        gamma = np.clip(0.0379 * np.log(exp_indices-1) + 0.7983, 0.5, 0.99)
                    
                    if observations[-1,-2] == 1:
                        gamma[0] = 1  
                    
                    for step in range(exp_len):
                        reward_sum = rewards[step] + gamma[step]*reward_sum
                        discounted_rewards[step] = reward_sum
                                                
                    discounted_rewards = np.flip(discounted_rewards)
                    
                    observations = np.hstack((observations,np.expand_dims(discounted_rewards, axis = 1)))
                    
                    lock.acquire()
                    if collect_obs.full():
                        collect_obs.get()
                    collect_obs.put(observations)
                    lock.release()
                    
                    observations = np.empty(state_n+3)
            
            # Update Counters to Track Progress
            lock.acquire()
            episod_counter.value += 1
            lock.release()
            
            print("Worker", worker, "Start Position", pos_start, "Steps", internal_step_counter)
            
            if internal_step_counter > internal_step_counter_best.value:
                internal_step_counter_best.value = internal_step_counter
                print("Worker", worker, "########################## Best Episod Length:", internal_step_counter)
            
            if internal_step_counter_best.value >= 50000:    
                print("\nREACHED GOAL of 50K Steps after", episod_counter.value, "episodes, in", time.time()-time_start, "seconds \n")  

            #time.sleep(1)
        
        print("Ending Worker:", worker)

if __name__ == "__main__":
       
    # Create Shared Variables
    lock = mp.Lock() # LOCK object to ensure only one Processes can access the locked object
    manager = mp.Manager() # MANGER object to create the shared variables
    collect_obs = manager.Queue(2) # Collector of Episods from EXEMPLARS
    #memory_buffer = manager.Queue() # Central Memmory Buffer -- implemented as numpy array within memorizer
    collect_examples = manager.Queue(2) # Collector of Training Examples
    episod_counter = manager.Value('i',0) # Number of Finalized Episodes
    step_counter = manager.Value('i',0) # Number of Finalized Steps
    internal_step_counter_best = manager.Value('i',0) # Length of the longest episod
    exemplar_model = manager.list()

    # Set Parallel Processes
    env_processes = []
    
    cores = mp.cpu_count()
    #cores = 3 # -1 as presumably 1 mp.manager takes over one process/core
    
    time_start = time.time()
    for worker in range(cores):
        p = mp.Process(target=LMxE, args=(worker,
                                          lock,
                                          collect_obs,
                                          collect_examples,
                                          episod_counter,
                                          step_counter,
                                          exemplar_model,
                                          internal_step_counter_best,
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
