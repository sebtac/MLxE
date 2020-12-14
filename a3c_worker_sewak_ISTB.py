""" A3C in Code - A3C Worker

A3C Code as in the book Deep Reinforcement Learning, Chapter 12.

Runtime: Python 3.6.5
Dependencies: numpy, matplotlib, tensorflow (/ tensorflow-gpu), gym
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""
import logging
import time
# making general imports
import threading
import os
import numpy as np
# making deep learning and env related imports
import tensorflow as tf
import gym
import math
# making imports of custom modules
from experience_replay_sewak import SequentialDequeMemory
from actorcritic_model_sewak import ActorCriticModel
# Configuring logging and Creating logger, setting the log to streaming, and level as DEBUG
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class A3C_Worker(threading.Thread):
    """A3C Worker Class

        A3C Worker implemented as a thread (extends threading.Thread). The function computes the gradient of the policy
        and value networks' updates and then update the global network parameters of a similar policy and value
        networks after every some steps or after completion of a worker's episode.

    """
    global_constant_max_episodes_across_all_workers = 500
    global_constant_total_steps_before_sync_for_any_workers = 64 # 128
    global_shared_best_episode_score = 0
    global_shared_total_episodes_across_all_workers = 0
    global_shared_total_steps_across_all_workers = 0
    global_shared_semaphore = threading.Lock()
    global_shared_training_stats = []
    global_shared_episode_reward = 0
    global__step_count_per_run = []
    memory = SequentialDequeMemory()
    worker_execution_counter = np.empty(0)
    new_alpha = 0.0
    global_constant_max = 0
    model_update_counter = 0
    K50_marker = 0
    
    def __init__(self, central_a3c_model, optimizer, worker_id, env_name, model_dir, workers_num, learning_rate, discounting_factor=0.98):
        """Initialize the A3C worker instance

            Args:
                central_a3c_model (ActorCriticModel): An instance of the ActorCriticModel or similar model shared by the
                                                    A3C master -- THE SHARED/TRAINABLE WEIGHTS!!!
                optimizer (tf.train.Optimizer): An instance of the Optimizer object as used in the A3C_Master to update
                                                    its network parameters.
                worker_id (int): An integer representing the id of the instantiated worker.
                model_dir (str): dir for saving the model. Should be the same location from where the A3C_Master will
                                                    retrieve the trained model for playing.
                discounting_factor (float): Value of gamma, the discounting factor for future rewards.
        """
        super(A3C_Worker, self).__init__()
        self.optimizer = optimizer
        self.worker_id = worker_id
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.central_a3c_model = central_a3c_model # the SHARED model/weights to be trained!!!
        self.worker_model = self.central_a3c_model       
        self.memory = SequentialDequeMemory(queue_capacity = workers_num * 1024)
        self.model_dir = model_dir
        self.this_episode_loss = 0
        self.this_episode_steps = 0
        self.this_episode_reward = 0
        self.this_episode_discountedreward = 0
        self.total_steps = 0
        self.steps_since_last_sync = 0
        logger.debug("Instantiating env for worker id: {}".format(self.worker_id))
        self.alpha = learning_rate 
        self.alpha_power = 0.998 #0.998 bast so far
        self.alpha_limit = 0.000001
        self.gamma = 0.97 # Discout rate! #0.97 bast so far
        self.reward_negative = -10.0 #-10.0 bast so far
        self.workers_num = workers_num - 1 # exclude 1 for LEARNER or 2 for LEARNER and MEMORIZER
        
        A3C_Worker.worker_execution_counter = np.zeros(workers_num)

    def run(self):

        if self.worker_id > 0:
            time_start = time.time()
            logger.debug("Starting WORKER_{}".format(self.worker_id))
            
            prob_adverse_state_initial = 1/5
            prob_adverse_state_type_multiplier = 0.0
            
            while A3C_Worker.global_shared_total_episodes_across_all_workers < A3C_Worker.global_constant_max_episodes_across_all_workers and A3C_Worker.K50_marker >= 0:

                while A3C_Worker.worker_execution_counter[self.worker_id-1] == 1:
                    #print("Waiting for Learner", A3C_Worker.worker_execution_counter, "Worker:", self.worker_id)
                    time.sleep(1)
                    #pass
                
                if A3C_Worker.worker_execution_counter[self.worker_id-1] == 0:
                    self.worker_model.set_weights(self.central_a3c_model.get_weights())
                
                    # ST: Adjust the learinig rate
                    
                    A3C_Worker.new_alpha = self.alpha*np.power(self.alpha_power,A3C_Worker.global_shared_total_episodes_across_all_workers)  
                    if A3C_Worker.new_alpha < self.alpha_limit:
                        A3C_Worker.new_alpha = self.alpha_limit
                    
                    self.optimizer.learning_rate = self.new_alpha
                    
                    states = []
                    actions = []
                    rewards = []
                    
                    done = False
                    # Collect Examples & Save them in the Central Observation Repository
                    current_state = self.env.reset()
            
                    # ENSURE EXPLORATION OF ADVERSE STATES
                    if A3C_Worker.global_shared_total_episodes_across_all_workers <= 1:
                        prob_adverse_state = prob_adverse_state_initial
                    else:
                        prob_adverse_state = np.clip(prob_adverse_state_initial/math.log(A3C_Worker.global_shared_total_episodes_across_all_workers,5), 0.05, 0.2)
            
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
                        #current_state[4] = current_state[0]*current_state[0]
                    if pos_start == 1 or pos_start == 6: 
                        current_state[0] = 1.5 # -2.4 MAX
                        #current_state[4] = current_state[0]*current_state[0]
                    if pos_start == 3:
                        current_state[2] = -0.150 #-0.0.20943951023931953 MIN
                    if pos_start == 4:
                        current_state[2] = 0.150 #0.0.20943951023931953 MAX
            
                    self.env.state = current_state
                    
                    initial_state = current_state

                    # Custom State Representation Adjustment to help agent learn to be closer to the center
                    current_state = np.append(current_state,current_state[0]*current_state[0]) 
            
                    while not done and self.this_episode_steps < 50000:
                        
                        # UPDATE GLOBAL COUNTERS
                        self.total_steps += 1
                        self.steps_since_last_sync += 1
                        self.this_episode_steps += 1
                        
                        # CREATE EXAMPLES
                        #policy_logits, values = self.worker_model(tf.convert_to_tensor(np.random.random((1, self.n_states)), dtype=tf.float32))
                        policy_logits, values = self.worker_model(tf.convert_to_tensor(np.array(np.expand_dims(current_state,axis=0)), dtype=tf.float32)) # ST to base the new step on the action of the previous one
                        stochastic_action_probabilities = tf.nn.softmax(policy_logits)
                        #print("1", self.n_actions, stochastic_action_probabilities.numpy()[0])
                        action = np.random.choice(self.n_actions, p=stochastic_action_probabilities.numpy()[0])
                        new_state, reward, done, _ = self.env.step(action)
                        
                        # Custom State Representation Adjustment to help agent learn to be closer to the center
                        new_state = np.append(new_state,new_state[0]*new_state[0])

                        # ST add desired behaviour to the reward fucntion
                        R_pos = 1*(1-np.abs(new_state[0])/2.4) # 2.4 max value ### !!! in documentation says 4.8 but failes beyound 2.4
                        R_ang = 1*(1-np.abs(new_state[2])/0.20943951023931953) ### !!! in documentation says 0.418 max value
                        reward = reward + R_pos + R_ang
                        
                        # Custom Fail Reward to speed up Learning of conseqences of being in adverse position
                        if done: 
                            reward = self.reward_negative # ST Original -1
                        
                        self.this_episode_reward += reward

                        states.append(current_state)
                        actions.append(action)
                        rewards.append(reward)
                        
                        current_state = new_state # ST - I think it was missing 
        
                    reward_sum = 0

                    discounted_rewards = []
                    gammas = []
                    step = -1                               
                    for reward in rewards[::-1]:  # reverse buffer r
                        step += 1
                        if step == 0:
                            self.gamma = 1
                            gammas.append(self.gamma)
                        else:
                            if done == False:
                                self.gamma = 0.99
                                gammas.append(self.gamma)
                            else:
                                self.gamma = np.clip(0.0379*np.log(step) + 0.7983, 0.5,0.99)
                                gammas.append(self.gamma)
                        reward_sum = reward + self.gamma * reward_sum
                        discounted_rewards.append(reward_sum)
                    discounted_rewards.reverse()
                    self.this_episode_discountedreward=np.float(discounted_rewards[0])

                    cut_off = 1024

                    A3C_Worker.memory.store(states[-cut_off:], actions[-cut_off:], discounted_rewards[-cut_off:])
                    
                    A3C_Worker.global_shared_training_stats.append((self.worker_id, 
                                                                    A3C_Worker.global_shared_total_episodes_across_all_workers,
                                                                    self.this_episode_steps, 
                                                                    self.this_episode_reward, 
                                                                    self.this_episode_discountedreward, 
                                                                    self.this_episode_loss))
                    
                    A3C_Worker.global__step_count_per_run.append(self.this_episode_steps)
                    with A3C_Worker.global_shared_semaphore:
                        A3C_Worker.global_shared_total_steps_across_all_workers += self.total_steps
                        A3C_Worker.global_shared_total_episodes_across_all_workers += 1
                    self.total_steps = 0
                
                    A3C_Worker.worker_execution_counter[self.worker_id-1] = 1

                    #; Initial_State {}; Current_State {}
                    logger.info("Ending episode {}/{}; worker {}; Init_Cond {}; Steps {}; Done {}".format(
                        A3C_Worker.global_shared_total_episodes_across_all_workers,
                        A3C_Worker.global_constant_max_episodes_across_all_workers, 
                        self.worker_id,
                        pos_start,
                        self.this_episode_steps,
                        done))
                        #initial_state,
                        #current_state))
                    
                    if self.this_episode_steps >= 50000:
                        A3C_Worker.K50_marker = 1
                        print("\nREACHED GOAL of 50K Steps in", A3C_Worker.model_update_counter , 
                              "Learning Iterations after", A3C_Worker.global_shared_total_episodes_across_all_workers, 
                              "games, in", time.time()-time_start, "seconds.")

                    self.this_episode_steps = 0

        if self.worker_id == 0:
            logger.debug("Starting WORKER_{}".format(self.worker_id))
            while A3C_Worker.global_shared_total_episodes_across_all_workers < A3C_Worker.global_constant_max_episodes_across_all_workers and A3C_Worker.K50_marker >= 0:
                self.worker_model.set_weights(self.central_a3c_model.get_weights())
                while np.sum(A3C_Worker.worker_execution_counter) < self.workers_num:
                    time.sleep(1)
                    #pass

                if np.sum(A3C_Worker.worker_execution_counter) == self.workers_num:
                    
                    if A3C_Worker.global_shared_total_steps_across_all_workers > A3C_Worker.global_shared_best_episode_score:
                            A3C_Worker.global_shared_best_episode_score = A3C_Worker.global_shared_total_steps_across_all_workers    
                            self._update_best_model(A3C_Worker.global_shared_total_steps_across_all_workers)
                            
                    print("###### ###### ###### ###### Ending after:", 
                          A3C_Worker.global_shared_total_steps_across_all_workers , 
                          "steps; Best Run:", 
                          A3C_Worker.global_shared_best_episode_score, 
                          "Steps;", A3C_Worker.new_alpha,"#####")
                            
                    self._sync_worker_gradient_updates_with_global_model()
                    

                    A3C_Worker.worker_execution_counter = np.zeros(self.workers_num)
                
                with A3C_Worker.global_shared_semaphore:
                    A3C_Worker.global_shared_total_steps_across_all_workers = 0
                    
    @classmethod
    def reset(cls):
        print("Resetting the GlobalSharedVariables")
        A3C_Worker.global_shared_total_episodes_across_all_workers = 0
        A3C_Worker.global_shared_best_episode_score = 0
        A3C_Worker.global_shared_training_stats = []
        #A3C_Worker.memory.clear()
        
    def _update_best_model(self,mem_len):
        """Rewrite the saved model with a beteer performing one

            This function rewrites the existing model (if any) saved in the model_dir, if any worker thread happens
            to obtain a better score in any of the episodes than the laste best score for an episode by any of the
            workers.

        """
        #A3C_Worker.global_shared_best_episode_score = self.this_episode_reward
        with A3C_Worker.global_shared_semaphore:
            logger.info("### Saving best model - worker:{}, episode-steps:{} ###".format(self.worker_id,mem_len))
            
            self.central_a3c_model.save_weights(os.path.join(self.model_dir, 'model_{}_global.h5'.format(self.env_name)))
        # ST - added as i think that saving the central model is not good as it can be different than the one used to produce the best results.
        #self.worker_model.save_weights(os.path.join(self.model_dir, 'model_{}.h5'.format(self.env_name)))

    def _reset_episode_stats(self):
        """Internal helper function to reset the episodal statistics
        """
        self.this_episode_steps = 0
        self.this_episode_loss = 0
        self.this_episode_reward = 0
        self.this_episode_discountedreward = 0
        #self.memory.clear()
        return self.env.reset()

    def _sync_worker_gradient_updates_with_global_model(self):
        """Internal helper function to sync the gradient updates of the worker with the master

            This function is called whenever either an episodes ends or a pecified number of steps have elapsed since
            a particular worker synced with the master.
            In this process the losses for the policy and values are computed and the loss function is differentiated
            to fund the gradient. The so obtained gradient is used to update the weights of the master (global network)
            model parameters. Then the worker copies the updated weights of the master and resumes training.

        """        
        mem_len = A3C_Worker.memory.get_memory_size()
        batch = np.minimum(A3C_Worker.global_constant_total_steps_before_sync_for_any_workers, mem_len)
        runs = mem_len // batch + 1
        
        for i in range(runs):
            momory_extract = A3C_Worker.memory.extract(batch_size = batch)

            states_input = momory_extract[0]
            rewards_input = momory_extract[2]
            actions_input = momory_extract[1]

            if len(rewards_input) == 0:
                pass
            else:
                with tf.GradientTape() as tape:
                    total_loss = self._compute_loss(states_input,rewards_input,actions_input)
                self.this_episode_loss += total_loss
                # Calculate local gradients
                grads = tape.gradient(total_loss, self.worker_model.trainable_weights)
                # Push local gradients to global model
                self.optimizer.apply_gradients(zip(grads, self.central_a3c_model.trainable_weights))
                # Update local model with new weights
                #self.worker_model.set_weights(self.central_a3c_model.get_weights())

        A3C_Worker.model_update_counter += runs

        self.steps_since_last_sync = 0

    def _compute_loss(self,states,rewards,actions):
        """Function to compute the loss

            This method compute the loss as required by the _sync_worker_gradient_updates_with_global_model
            method to compute the gradients

        """
        logits, values = self.worker_model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(rewards)[:, None], dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2 # this is a term to be minimized in trainig
        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits), [-1,1])
        policy_loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits), [-1,1])
        # the next is the equation (Gradient(log(PI(A|S,THeta)*A))) that needs to be minimized
        policy_loss *= tf.stop_gradient(advantage) # advantage will be exluded from computation of the gradient; thsi allows to treat the values as constants
        policy_loss -= 0.01 * entropy # NOT SURE WHY THIS IS DONE -- Seems not standard approach
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss)) # NOT SHURE WHY IT IS SO -- Might be not standard implementation
        return total_loss

if __name__ == "__main__":
    raise NotImplementedError("This class needs to be imported and instantiated from a Reinforcement Learning "
                              "agent class and does not contain any invokable code in the main function")