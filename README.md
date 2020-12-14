# MLxE: Highly Effcient Architecture for Implementation of Reinforcement Learning Algorithms

# Objectives of the Project:
To intorduce a highly efficient implementation of the Reinforcement Learning (RL) algorithms that takes adventage of both the recent algorithmic developments in the field of RL and maximises the utilization of hardware capabilities available to the RL researcher or architect. This efficiency is achieved via employeeing the following three proposiotions:

1.) MLxE Architecture, a highly efficient, Multiprocessing based implementation of RL algorithms. It achieves >2x efficiency improvement relative to Threading based implementations in the test environment.

2.) Update to the classical implemntations of A3C, RAINBOW and SEC algorithms with recent developments in the field of RL such as #########

3.) Addition of the "Task" specific modifications to the learning process to maximize the learining efficiency

We show the feasibility and effectiveness of the MLxE Architecture uning the "Task" of Gym's Cart-Pole Environment where we train the agent to play the environment for 50,000 steps after only 8 minutes of training, with only few hundreds game examples, on machine with 8-core CPU and no GPU.

We root our work in the code for the book Deep Reinforcemenrt Learning by Mohit Sewak and refer to it as "The Book" in the text below. Its github reporsitory is: https://github.com/mohitsewak/DeepReinforcementLearning

# MLxE Architecture
MLxE Architecture is designed for RL learning and research applications with focus on ease of understanding and of modifying underlying RL algorithms (A3C, Rainbow, SAC,...) thus it is implemented in linear fashion with limited use of python programmatical overhead (Classes, Functions,...). 

This is an TensorFlow 2 implementation making it one of the most up to date implementations of the RL Algorithms

At its core, MLxE Architecture consist of multiple processes (called Memorizer, Learner and Executors) executed (optimally) in parallel, each with its deditacted function:

M Process - the Memorizer: Manages the Memory of the Learining Agent. It collects examples generated by Executors, modifies them and repackages them for consumption by the Learner

L Process - the Learner: The core proccess of the Learing Agent responsible for updating model weights through a SGD based algorithm. It consumes the trainig examples prepared by Memorizer and sends updated models to Executors for generation of the next wave of the examples.

xE Processes - the Executors: Generate new examples for trainig utilizing the model computed by the Learner. There are as many (x) executors as there are cores on the machine you are using, thus "xE" in the name of the archtecture. This number can be reduced by 2 if the Memorizer and the Learner are imlemented in their own processes. For example, this project was created using the ML6E and ML8E architectures since our machine had 8 CPU cores. 

You can execute the Learner's calcualtions on GPU as well but still one CPU core will be dedicated to the process handling the Learner. 

There are four sub-architectures tested. We are comparing their performce and make recomendation for the best performing one: the Iterative-Synchronous MLxE:

1.) Iterative-Synchronious Threading Based (ISTB) - it is our banchark implementation based on modification of a code available in the book.
- Iterative - learning phase is perfomed intermittently with the example generation phase
- Synchronious - each executor generates only one example per iteration, waiting idle for closing of all executors and update of the model in given iteration
- Model update for the Learner and Executors is performed at the end of each iteration

2.) Iterative-Synchronious MLxE Based (IS MLxE) - the best performing architecture, used in final implementations of the RL Algorithms
- Implemented with Multiprocessing
- Other details are the same as that of the ISTB architecutre
- Due to iterative nature of the architecture, all cores are used as Executors in Example Generation Phase and one process is used subsequently for combined Memorization and Learing Phases 

3.) Iterative-Asynchronious MLxE Based (IA MLxE) - During Example Generation Phase all Executors generate examples till each Executor generates at least one Example. In other words, Executors start generation of another example if they finish their previous example before all Executors fnished generating their first example in given iteration.

4.) On-Line-Asynchronious MLxE Based (OA MLxE) - the Memorizing and Learing Phases are spawned in their own processes and are run in parallel to the Executors. The model is continously updated and its new version becomes imidiately available to Executors for initialization of the next Example Generation.

# Algorithmic Updates

This project implemented the follwing RL Algorithms:
1.) A3C
2.) Rainbow
3.) SAC

Sunsequently, we have updated their peper based implementations with recent methodological developments in the field of RL, mainly:
1.)############
2.)###########
3.)#############

We dicuss the impact of such improvements on our test environment.

# "Task" Specific Modifications

The "Task" for this project is the GYM's implementation of the Cart Pole Environment. It is the simplest, and thus the fastest, environment to work with that allows for rapid feedback while you learn the existing or explore new approachses to RL. It is also a flexible environment that allows you to "raise the bar" for your agent with few code modifications. In its base implementation, your agent is expected to keep the pole upright for 200 moves of the cart. We, instead, aim to train an agent that can do so indifinatively (in theory at least) and will learn such level of proficiency on a sample of just couple houndreds of games.

In short, cart-pole-v0 enviroment requires the agent to balance a pole on a moving cart by controling the movement of the cart (left or right). The agent gets positive reward of 1 for every move of the cart as long as the pole is standing relatively upward and by not going beyound the edges of the plane on which the cart moves (think of a table). Once either of the failure conditions is reached, the agent gets reward of 0 and the game is reset back to a random inital position. The state is represented with an array(cart's position, cart's velocity, pole's angle, pole's angular-velocity) and there are only two actions for the cart (move-left, move-right). Note that there is no "Do Nothing" action; the agent must move in each step. 

One of the objectives in the development of new Reinforcement Learning algorithms is to design algorithms that can learn usefull models accross wide range of environments without modification in the set of parameters. While this is a noble objective from the theoretical perspective, it is a limiting factor in real life applications. We should not accept a model for driving a car just becouse it is also capable of flying us to the Space Station. We should also be open to modifications to the learning algorithms and/or to their implementations that with addition of only couple of lines of code will lead to singificant speedup in learning (although without introducing changes to the nature of the environment).

Another motivation for intoduction of the "Task" Specific Modifications is to allow the agent to learn a "proper" behaviour. In many implementations, just achieving the goal of the task is enough to call it a success. In others, the way that such goal is achieved matters as much. For example, you want your car to drive on the streets following certain road rules and not just to do it as efficantly as a racing driver would. In the case of the Cart Pole environment, we have traind a lot of "successfull" agents that were able to maintain upright the pole for 10s of thousands of cart moves. But they were doing it in an "akward" way where the pole was almost not moving at all or it was stabilized close to the edges of the plane on which the cart moves. Both behaviours are "not-human-like" in a sense that we expect human to play the game trying to stabilize the pole in the middle of the plane and to control it not in a such stiff manner. Thus, we have introduced the following "Task" Specific Modifications:

1.) Modification of the State Representation
* Addition of the squared cart position (env.state[0]) - this allows the agent to learn quicker that being close to the center is good for avoiding running out of plane and helps it behave more "naturally".

2.) Advarse Modification of the Initial Position Parameters
* Addition of a randomly selected advarse initial conditions that put the cart-pole close to the edge of the plane or leaning to the side more heavilly at the begginig of the game - the agent intializes each game by drawing random values for the four state parameters (close to 0 each). Advarse initialization additionaly modifies randomly one of those parameters. The probability of the advarse initialization decreases as trainig progresses similarly to the monotonic deacres in larning rate of the optimizer. This allows the trainig agent to experience more freaqently advarse conditions, states with much higher informative value than non-advarse states (see the next point). 

3.) Modifying the Reward - Note that only the fail condition results in a true learning feedback (no reward). All other states return value of 1, thus the agent cannot know that leaning can be bad until it reaches the threashold angle beyound which the game ends. Thus we have introdcued two modification to the reward
* Failure "Reward" set to -10 (instead of 0) - to provide the agent with stronger "incentive" to avoid failure conditions. What truly matters here is a magnitude of the difference betwen the "Negative-State" and "Positive-State" Rewards. We should achive comperable results by increasing the Positive-State Reward.
* Introdcution of non-constant reward discount factor (the gamma coeffcient) - to measure the value of the state or of the action, RL uses cumulative rewards which are a sum of the imidiate reward in given state and/or for action just taken and the expected sum of future rewards after following such move. Note that in environments with infite horizons (i.e. where the game can last indifinativly and cart-pole is a such game!) the expected cumulative reward can be infinite as well. Infinities are not friendly to RL algorithms thus, we use a discount factor (DF) in front of the Expected Sum of Future Rewards term to limit that value. The DF value is between 0-1, usually close to 1 like .99,.95 or.90. The selection of that value has a practical/conceptual meaning as well as it defines the horizon within which we accumulate the future rewards. The closer the value of DF is to 1 the further ahead we attribute the effects of the current actions. Our attention span increases in a way. We theoretize that in the case of the cartpole attention span of the agent should differ depending on the "saftey" of the state the agent is in. If it is a relatively safe state the agent can think about returns in a long time span with gamma close to 1. If it is a relaively dengerous state, the agent should focus more on imidiate consequences of its actions with smaller values of gamma. Thus we have introduced an algorithm that modifies the value of the discount factor depending on how cloese given state is to the failure condition.
* Introduction of additianal rewards representing a realtive safety of the state. Our agents get additional one point for the cart being at the center of the plane and one for the pole being in perfectly vertical possition. Those rewards are proportional to the current distance of the cart and that of the pole from the center and the vertical positions. Thus the agent can get up to 3 points in each state but if the cart or pole will be at the threshold of failing, the agent will achive reward of 0 for the corresponding reward component. 
* The combined effect of the three above Reward Function modifications is that the reward assigned to the states increases monotonically in logarithmic fashion as the state is farther away from the failure condition allowing the agent to get more precise sense of the value of the state it is in.

Here we need to say a word of caution. Such adjustments to the environment can have adverse effects on the quality of the learnt model. The effects here are akin to overspecification of the regular Machine Learning models and might express themsleves by lack of generalization by our learnt model. In other words, if we train the agent toward expressing certain behavior, it might not perform well in real life if the real-life conditions are different from those on which the agent was trained. Had we only emphasised the moving out of plane condition in the training, the agent might have had hard time learning to avoid the pole-failing condition. In the car driving analogy, the agent trained for driving on a highway might not perform well on city streets. But we theoretize that in more complex environments, we could use this familiy of adjustments to teach the agent wide variety of skills in a sequence, focusing on the general skills at the begginig of the training and than expose the agent to a variety of different (but specific) conditions in later iterations of the training. This would be akin to Transfer Learning in the text and vision domains of Deep Learinig.

# Development Details

## Exploring A3C Model

We have started our exploration by utilizing the A3C algorithm's implementation in the book as a initial benchmark and performed the follownig steps:

1.) Fixed issues with the code (detailed comments in the code):
- Calculation of Policy_Loss
- Utilization of the actual steps taken by the agent in example generation (as opposed to the random search in the original code)

This became our base implementation

Agent Implemented in file: a3c_worker_sewak_base.py

2.) Implemented Deep Learning Specific Adjustments to the Model and Hyper-Parameters:
- Increased the batch size to 64
- Added monotonic decrease in Learing Rate relative to the number of episodes run with:
    self.alpha_power = 0.998
    self.alpha_limit = 0.000001
- Introduced Broken-Dimond-Shaped Model Design and increased the network size: common_network_size=[128,258,512], policy_network_size=[256,128,64], value_network_size=[256,128,64]
- Changed the Optimizer to RectifiedAdam -- requaires tensorflow_addons
- Changed Gamma coeffcient to 0.97

Agent Implemented in file: a3c_worker_sewak_DNN_Adjusted.py

3.) Implemented the "Task" Specific Modifications dicussed above

Agent Implemented in file: a3c_worker_sewak_Task_Modifications.py

The above changes are additive.

The comperative performance analysis shows that the Base-A3C implementation is very unstable. Although, it achieves ability to execute 10K steps in a single game, this level of performance is not maintained over time. In itself, this is not suprising as RL training is inherently unstable. Contrary to other families of machine learing models, we are presenting the learing agent with subset of possible condintions at any given time so its performace must decrease as the agent explores new areas of the state space. But it also fails to gain a visible learing trend in a long run suggesting it does not accumulate well the skills gained earlier in the training. Modification of Deep Learning Hyper-Parameters stabilizes training and allows the agent to generate visible learning trend. But such agent is able only to play a game for up to 1000 steps. Ultimately, introduction of the "Task" Specific modifications, results in the agent's ability to achieve 50K steps per game with visible, although somewhat unstable, training trend. 

Note that the initial performance of the agent with "Task" Specific Modifications is worse than that of the the other two agents. This has to do with intorductionso of random number of initial states with advarse characteristics and more so at the beggining of the training. But ultimately the agent learns to deal with those situations and maintains its skills in later stages of the training bulding the basis for even stronger performance later in the training.

The below chart depicts three runs of the learning process, one for each type of the agent. It is shown on logarithmic scale to allow detailed comparison of the agents performace for games with lower number of steps. The lines represent 8-game moving averages to expose the trends in the results, if any. More runs were performed for each scenario and while individual runs might have differed from those included in the chart they featured the general characteristics as discussed in the above paragraphs.

![github-small](https://github.com/sebtac/MLxE/blob/main/Sewak%20-%20Models%20Comparison%20-%208-Step%20MA.jpg)

The implemented modifications in the architecure and the algorithm resulted in desired performance improvements nevertheless the level of agent's performacne, its stability and reproductability is still far from what can be expected in such small and uncomplicated task as Cart-Pole.

## ISTB Architecture

Detailed analysis of the code showed that the A3C implementation in the book suffers from two issues that we theorethise to contribute to the training instability observed above. 

1.) Lack of true memory buffer - the code resets the momory every time new apisode is initiated and when an update to the model is made (every 10 game-steps in the book's implementation and 64 game-steps in ours). Thus the effective buffer size is maximally 64 and often less so when threads clean the memory in close succession. The variable "batch size" can congribute to the training instability.

2.) The model updates are performed with unstable target. Only the central model is being updated with gradients but the gradients are caluclated using model currently used by given worker. Since workers make model updates intermitently "whenever they are read", it introduces "confusion" as to what is the goal of the training.

To address the above issues, we have further modified the A3C code, mainly:

1.) Implemented a MLxE algorithm consiting of:
- Process 0 assigned to Learner
- All remainig Processes assigned to Executors
- Memorizer's function split between Executors and the Learner
- Iterative-Synchronized Architecure:
    - First Executors generate examples and update Memory Buffer than Learner samples memory buffer and performs model updates and shares the resultsing model with the Executor       for the use in the next iteration of example generation
    - All executors generate only one set of examples per iteration
    - Learner performs as many model updates per iteration as there are possible unique batches in the memory buffer (default batch size is 64 thus with full memory buffer there       will be 112 batches sampled)
- The Targer Model in a model update is held constant and set to the one used in generation of the examples in the current Iteration. Keeping target model constant is a well established practice to reduce a variance of the estimates and thus making the model training process more stable. The intiuite explonations of such aproach is that by heaving the target model constant, all model updates aim to improve the models ability to match the given target. Were we changing the target our trainig process woudl be "chasing" different goal at each iteration of the model update making the training process more unstable.
    
2.) Implemented the complete Memory Buffer:
- there is one memory buffer, shared accross all workers
- it is implementated with deque from python's collections module
- the capacity of the Queue (maxlen) is 1024 * Number of Executors thus for the ML7E Thread Based implementation the Memory Buffer is holding up to 7168 cases collected from all Executors and possibly over multiple iterations
- when the full capacity of the Queue is reached the oldest entries are removed and are substituted with the newest ones
- training cases are sampled from the memory buffer as many times as there are possible unique batches in the memory buffer
- when given episod generates more than 1024 examples it saves to the buffer only the last 1024 examples (those the closest to the end state thus the most informative)

3.) Changed the progress measure to the sum of examples generated by all Executors in given iteration. This is so to base the assement of the modle perfomance on more stable metric that is less sensitive to overly optimistic or negative episods.

The ISTB Architecure resulted in significant improvement of the traing process and the agent gaint gaide an ability to play the game for 50K steps after exploring just 350 games. When left learing till 500 games the agent was returinig to the 50K steps level many times with ever higher freqency as the leraning continued. The below chart shows the comparison of the runs for the two approaches: 

![github-small](https://github.com/sebtac/MLxE/blob/main/Sewak%20Task%20Modified%20vs.%20ISTB%20Comparison%20-%208-Step%20MA.jpg)

We, actually, stopped the learining at 500 episodes due to the time it was taking the 8 Thread, 8 CPU cores based implementation of ISTB architecture - 2 hours for the 500 games run. We contribute the performce improvement mostly to the introduction of the constant Target Model but not completelty. When the Target Model was allowed to be updated to the Central Model continously, it still was traing to the 50K level occassionally in similar time span but featured freaquently runs where it did not converge well within 3000 games. Thus we theroretize that two additional factors are responsibel for such perfomarce. First, the fact that also most examples were generated with the arget model (at least later in the trainig) and that the number of model updates has increased in given iteration (to 112 per iteration) as well as to the the higher Batch Size of 64 (was 10).

Nevertheless, the are still two factors that introduce inefficency in the current implementation: Thread Based implementation and the iterative-synchronious nature of the archirtecture.

## MLxE Architecture









