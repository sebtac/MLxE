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
    - Combined Memorizer and Learner executed iteratively with Executors

Test:
    - Priority Replay Buffer
    - Three-Factors:
        - Age (Inverse-Proportional)
        - Safety (Inverse-Proportional)
        - TD Error (Proportional)   
    - best performace with Prioirty Weights based on all three factors.
        
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

        # GYM Settings
        "max_episodes": 1000, # How many Episodes do you want to run?
        "env_name": 'CartPole-v0', # the name of the GYM Environment to run
        
        # MODEL Definition
        "state_n": 4, # value of env.observation_space() (*2 as current_state and next_state)
        "state_n_adj": 1, # all Task Specific Adjustments (*2 as current_state and next_state)
        "state_n_add": 4, # all additional step descriptions (Reward, Done, Action, Discounted Reward)
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
        "experience_max_batch_size": 256, # Maximum number of cases to save to Memory Baffer from each experience (most recent) (tested 1024-64: 256 Best)
        
        # advarse STATE Probability
        "prob_advarse_state_initial": 0.2, # Probability of choosing advarse case when starting new episode
        "prob_advarse_state_type_multiplier": 0.0, #Adjustment to the Probability to control the ratio of issue types that are promoted through advarse initialization
        
        # REWARD Incentives
        "reward_negative": -10.0, # Override the GYM's default negative return
        
        # PRIORITY MEMORY BUFFER
        "pmb_cols": 5,  #
        "pmb_alpha": 0.9, # Not Implemented                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8]
        "pmb_beta": 0.4, # Tested till 0.8 but best reuslts with 1.0 # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
        "pmb_beta_increment": 0.001, # Originally 0.001
        "pmb_td_error_margin": 0.01,                                # pi = |td_error| + margin
        "pmb_abs_td_error_upper": 1,
               
        }


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
    experience_max_batch_size = args["experience_max_batch_size"]
    state_n_adj = args["state_n_adj"]
    state_n_add = args["state_n_add"]
    pmb_cols = args["pmb_cols"]
    pmb_alpha= args["pmb_alpha"]
    pmb_beta = args["pmb_beta"]
    pmb_beta_increment = args["pmb_beta_increment"]
    pmb_td_error_margin = args["pmb_td_error_margin"]
    pmb_abs_td_error_upper = args["pmb_abs_td_error_upper"]
        
        
    # Assign EXECUTOR to all workers
    if worker >= 0:
        
        time_start = time.time()
        print("Starting E:", worker)
        
        # Establish Environment
        env = gym.make(env_name).unwrapped #unwrapped to access the behind the scenes elements of the environment
        
        # Define A3C Model for Executors
        inputs_executor = tf.keras.Input(shape=(state_n+state_n_adj,))        
        
        common_network_executor = Dense(common_layers_n[0], activation='relu')(inputs_executor)
        common_network_executor = Dense(common_layers_n[1], activation='relu')(common_network_executor)
        common_network_executor = Dense(common_layers_n[2], activation='relu')(common_network_executor)
 
        policy_network_executor = Dense(policy_layers_n[0], activation='relu')(common_network_executor)
        policy_network_executor = Dense(policy_layers_n[1], activation='relu')(policy_network_executor)
        policy_network_executor = Dense(policy_layers_n[2], activation='relu')(policy_network_executor)
        
        value_network_executor = Dense(value_layers_n[0], activation='relu')(common_network_executor)
        value_network_executor = Dense(value_layers_n[1], activation='relu')(value_network_executor)
        value_network_executor = Dense(value_layers_n[2], activation='relu')(value_network_executor)
        
        logits_executor = Dense(action_n)(policy_network_executor)        
        values_executor = Dense(1)(value_network_executor)
        
        model_executor = Model(inputs=inputs_executor, outputs=[values_executor, logits_executor])
        
        # Define A3C Models for Learners        
        if worker == 0: #
            
            # Define BASE Model - Target
            inputs_base = tf.keras.Input(shape=(state_n+state_n_adj,))
            
            common_network_base = Dense(common_layers_n[0], activation='relu',name="1")(inputs_base)
            common_network_base = Dense(common_layers_n[1], activation='relu',name="2")(common_network_base)
            common_network_base = Dense(common_layers_n[2], activation='relu',name="3")(common_network_base)

            policy_network_base = Dense(policy_layers_n[0], activation='relu',name="7")(common_network_base)
            policy_network_base = Dense(policy_layers_n[1], activation='relu',name="8")(policy_network_base)
            policy_network_base = Dense(policy_layers_n[2], activation='relu',name="9")(policy_network_base)

            value_network_base = Dense(value_layers_n[0], activation='relu',name="4")(common_network_base)
            value_network_base = Dense(value_layers_n[1], activation='relu',name="5")(value_network_base)
            value_network_base = Dense(value_layers_n[2], activation='relu',name="6")(value_network_base)
            
            values_base = Dense(1,name="10")(value_network_base)            
            logits_base = Dense(action_n,name="11")(policy_network_base)

            model_base = Model(inputs=inputs_base, outputs=[values_base, logits_base])
            
            
            # Define MAIN Model - Trainable Model
            inputs_main = tf.keras.Input(shape=(state_n+state_n_adj,))
            common_network_main = Dense(common_layers_n[0], activation='relu')(inputs_main)
            common_network_main = Dense(common_layers_n[1], activation='relu')(common_network_main)
            common_network_main = Dense(common_layers_n[2], activation='relu')(common_network_main)

            policy_network_main = Dense(policy_layers_n[0], activation='relu')(common_network_main)
            policy_network_main = Dense(policy_layers_n[1], activation='relu')(policy_network_main)
            policy_network_main = Dense(policy_layers_n[2], activation='relu')(policy_network_main)
            
            value_network_main = Dense(value_layers_n[0], activation='relu')(common_network_main)
            value_network_main = Dense(value_layers_n[1], activation='relu')(value_network_main)
            value_network_main = Dense(value_layers_n[2], activation='relu')(value_network_main)

            logits_main = Dense(action_n)(policy_network_main)
            values_main = Dense(1)(value_network_main)

            model_main = Model(inputs=inputs_main, outputs=[values_main, logits_main])
            
            # Define Optimizer
            optimizer = tfa.optimizers.RectifiedAdam(lr_alpha)
            
            lock.acquire()
            executor_model.append(model_main.get_weights()) # the first call MUST be append to create the entry [0]
            print("Saved Model", worker, len(executor_model))
            lock.release()

            memory_buffer = np.full((state_n+state_n_adj) * 2 + state_n_add + pmb_cols,0.0)
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

            observations = np.empty((state_n+state_n_adj) * 2 + 3)
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
                        
                #current_observation = np.append(current_state,(reward, done, action))
                current_observation = np.append(current_state,next_state)
                current_observation = np.append(current_observation,(reward, done, action))

                observations = np.vstack((observations, current_observation))
                current_state = next_state
                internal_step_counter += 1
                
                if internal_step_counter == 1:
                    observations = observations[1:]
                
                if done == True or internal_step_counter == internal_step_counter_limit:
                    
                    observations = observations[-np.minimum(observations.shape[0], experience_max_batch_size):]
                    
                    exp_len = observations.shape[0]
                    exp_indices = np.array(range(exp_len)) + 1
                    rewards = np.flip(observations[:,(state_n+state_n_adj) * 2 ])
                    #print("rewards",rewards)
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
                    observations = np.hstack((observations,np.zeros((observations.shape[0],pmb_cols))))
                                                                
                    lock.acquire()
                    collect_obs.put(observations)
                    lock.release()
                    
                    observations = np.empty((state_n+state_n_adj) * 2 + 3)

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
                    
                    #print("memory_buffer",memory_buffer.shape)
                    #print("exp_temp",exp_temp.shape)
                    
                    memory_buffer = np.vstack((memory_buffer,exp_temp))
                    memory_buffer = memory_buffer[-np.minimum(memory_buffer.shape[0],experience_batch_size*executors_n):,:]

                # PRIORITY MEMORY BUFFER
                
                # Inverse Discounted Reward Probability
                #print("DR", memory_buffer[:,-(pmb_cols+1)])
                dr_min = np.min(memory_buffer[:,-(pmb_cols+1)])
                memory_buffer[:,-(pmb_cols)] = memory_buffer[:,-(pmb_cols+1)] - dr_min
                dr_max = np.max(memory_buffer[:,-(pmb_cols)])
                memory_buffer[:,-(pmb_cols)] = 1 - memory_buffer[:,-(pmb_cols)]/ dr_max + 0.01
                dr_sum = np.sum(memory_buffer[:,-(pmb_cols)])
                memory_buffer[:,-(pmb_cols)] = memory_buffer[:,-(pmb_cols)]/ dr_sum
                #dr_sum_check = np.sum(memory_buffer[:,-(pmb_cols)])
                #print("dr_sum_check",dr_sum_check)
                
                # Inverse "Age" Probability
                #print("Age", memory_buffer[:,-(pmb_cols-1)])
                memory_buffer[:,-(pmb_cols-1)] += 1
                age_max = np.max(memory_buffer[:,-(pmb_cols-1)])
                memory_buffer[:,-(pmb_cols-2)] = age_max - memory_buffer[:,-(pmb_cols-1)] + 1.0
                age_sum = np.sum(memory_buffer[:,-(pmb_cols-2)])
                memory_buffer[:,-(pmb_cols-2)] = memory_buffer[:,-(pmb_cols-2)] / age_sum
                             
                # Proportional TD Error Probability                
                #best_action_idxes, target_q = self.model.action_value(self.b_next_states)  # get actions through the current network
                # get td_targets of batch states
                target_q, target_logits = model_base(tf.convert_to_tensor(memory_buffer[:,5:10], dtype=tf.float32))
                #print("target_logits",target_logits)
                #print("target_q",target_q)
                #td_target = memory_buffer[:,10] + 0.9 * target_q * (1 - memory_buffer[:,11])
                td_target = np.expand_dims(memory_buffer[:,10],axis = -1) + 0.9 * target_q * np.expand_dims((1 - memory_buffer[:,11]),axis = -1)
                #print("td_target",td_target)
                predict_q, predict_logits = model_main(tf.convert_to_tensor(memory_buffer[:,:5], dtype=tf.float32))
                #print("predict_q",predict_q)
                #td_predict = predict_q[np.arange(predict_q.shape[0]), self.b_actions]
                abs_td_error = np.abs(td_target - predict_q) + pmb_td_error_margin
                #print("abs_td_error",abs_td_error)
                clipped_td_error = np.where(abs_td_error < pmb_abs_td_error_upper, abs_td_error, pmb_abs_td_error_upper)                
                #print("clipped_td_error",clipped_td_error)
                #td_error_sum = np.sum(clipped_td_error)
                #td_error_weight = clipped_td_error/td_error_sum
                #print("td_error_weight",td_error_weight)
                #print("Sum", np.sum(td_error_weight))
                #print(td_error_weight.shape)
                #print(type(td_error_weight))
                #memory_buffer[:,-(pmb_cols-3)] = td_error_weight[:,0]
                memory_buffer[:,-(pmb_cols-3)] = clipped_td_error[:,0]
                td_error_sum = np.sum(memory_buffer[:,-(pmb_cols-3)])
                memory_buffer[:,-(pmb_cols-3)] = memory_buffer[:,-(pmb_cols-3)] / td_error_sum
                #print("td_error_weight",td_error_weight)
                #print("td_error_weight MB", memory_buffer[:,-(pmb_cols-3)])
                
                #ps = np.power(clipped_error, self.alpha)
                
                memory_buffer[:,-1] = np.average(memory_buffer[:,[-(pmb_cols),-(pmb_cols-2),-(pmb_cols-3)]], axis = 1 ) # 320 but can fail to converge
                #memory_buffer[:,-1] = np.average(memory_buffer[:,[-(pmb_cols),-(pmb_cols-2)]], axis = 1 ) # Best! 280-310
                #memory_buffer[:,-1] = memory_buffer[:,-(pmb_cols)] # Not Good, training collapsed
                #memory_buffer[:,-1] = memory_buffer[:,-(pmb_cols-2)] # Good, 310 Games
                #memory_buffer[:,-1] = memory_buffer[:,-(pmb_cols-3)] # ???
                pmb_beta = min(1., pmb_beta + pmb_beta_increment*executors_n)
                memory_buffer[:,-1] = np.power(memory_buffer[:,-1],pmb_beta)
                total_error_sum = np.sum(memory_buffer[:,-1])
                memory_buffer[:,-1] = memory_buffer[:,-1] / total_error_sum
                
                prob_sum_check1 = np.sum(memory_buffer[:,-(pmb_cols)])
                prob_sum_check2 = np.sum(memory_buffer[:,-(pmb_cols-2)])
                prob_sum_check3 = np.sum(memory_buffer[:,-(pmb_cols-3)])
                prob_sum_check = np.sum(memory_buffer[:,-1])
                print("prob_sum_check",prob_sum_check,prob_sum_check1,prob_sum_check2,prob_sum_check3)
                
                batch_size_min = np.minimum(batch_size,memory_buffer.shape[0])
                runs = memory_buffer.shape[0] // np.minimum(memory_buffer.shape[0],batch_size_min) + 1
                
                for i in range(runs):

                    sample_index = np.random.choice(memory_buffer.shape[0],
                                                    np.minimum(memory_buffer.shape[0],batch_size_min),
                                                    p = memory_buffer[:,-1],
                                                    replace=False)
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
                        #print("CS", example[:,:5])
                        values, logits = model_base(tf.convert_to_tensor(example[:,:5], dtype=tf.float32))
                        #print("Disc Reward", example[:,-(pmb_cols+1)])
                        advantage = tf.convert_to_tensor(np.expand_dims(example[:,-(pmb_cols+1)],axis=1), dtype=tf.float32) - values
                        value_loss = advantage ** 2 # this is a term to be minimized in trainig 
                        policy = tf.nn.softmax(logits)
                        entropy = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits), [-1,1])
                        #print("Action", example[:,-(pmb_cols+2)])
                        policy_loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=list(example[:,-(pmb_cols+2)].astype(int)), logits=logits), [-1,1])            
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
    
    cores = mp.cpu_count()
    
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
