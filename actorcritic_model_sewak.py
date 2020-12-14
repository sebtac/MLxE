""" A3C in Code - The Deep Learning Model for the Approximators

A3C Code as in the book Deep Reinforcement Learning, Chapter 12.

Runtime: Python 3.6.5
Dependencies: numpy, matplotlib, tensorflow (/ tensorflow-gpu), gym
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

Adjusted: Sebastian Taciak

Introduction of Dimond-Shaped Model Layouts
- ActorCriticModel_Dimond
- ActorCriticModel_DoubleDimond
"""
import sys
import time
sys.path.append(r'C:\Users\surface\Documents\Python\RL\MLxE\Mohit Sewak RL\Mohit12_A3C')

# Making common imports
import logging
# Making tensorflow and keras(tensorflow instance of keras) imports for subclassing the model and define architecture.
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
# Configuring logging and Creating logger, setting the log to streaming, and level as DEBUG
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Enabling eager execution for tensorflow
#tf.enable_eager_execution()

class ActorCriticModel(keras.Model):
    """A3C Model

        This class is for the policy and value approximator model for the A3C
        Both the master and the all the workers use individual instances of the same model class.

        This variant of the model class extends the keras.model and use tf.Model Sub-Classing feature.

        In a sub-class(ed) model, advanced features like shared network, shared inputs and residual networks could
        be easily implemented. The network layers need to defined in the __init__() method, and then their connection
        as required in the forward pass needs to be defined in the call or __call__ method.

        TensorFlow eager execution needs to be enabled for this to work as desired.


        Arguments:
            n_action (int): Cardinality of the action_space
            common_network_size (list): Defines the number of neurons in different hidden layers of the common/
                                        shared layers
            policy_network_size (int): Number of neurons in the hidden layer specific to the policy_network
            value_network_size (int): Number of neurons in the hidden layer specific to the value_network

    """
    #def __init__(self, n_actions, common_network_size=[128,64], policy_network_size=32, value_network_size=32):
    def __init__(self, n_actions, common_network_size=[256,128], policy_network_size=64, value_network_size=64):
        super(ActorCriticModel, self).__init__()
        logger.info("Defining tf model with layers configuration as: {}, {}, {}".format(common_network_size,
                                                                            policy_network_size, value_network_size))
        self.action_size = n_actions
        self.common_hidden_1 = layers.Dense(common_network_size[0], activation='relu')
        self.common_hidden_2 = layers.Dense(common_network_size[1], activation='relu')
        self.policy_hidden = layers.Dense(policy_network_size, activation='relu')
        self.values_hidden = layers.Dense(value_network_size, activation='relu')
        self.policy_logits = layers.Dense(n_actions)
        self.values = layers.Dense(1)

    def call(self, inputs):
        """Forward pass for the actorcritic model

            The call function is the wrapper on Python's __call__ magic function that is called when a class object is
            directly called without a specific method name.

            The Sub-Classing use this method to implement the forward pass logic.

        """
        # Forward pass
        common = self.common_hidden_1(inputs)
        common = self.common_hidden_2(common)
        policy_network = self.policy_hidden(common)
        logits = self.policy_logits(policy_network)
        value_network = self.values_hidden(common)
        values = self.values(value_network)
        return logits, values

class ActorCriticModel_Dimond(keras.Model):
    """A3C Model

        This class is for the policy and value approximator model for the A3C
        Both the master and the all the workers use individual instances of the same model class.

        This variant of the model class extends the keras.model and use tf.Model Sub-Classing feature.

        In a sub-class(ed) model, advanced features like shared network, shared inputs and residual networks could
        be easily implemented. The network layers need to defined in the __init__() method, and then their connection
        as required in the forward pass needs to be defined in the call or __call__ method.

        TensorFlow eager execution needs to be enabled for this to work as desired.


        Arguments:
            n_action (int): Cardinality of the action_space
            common_network_size (list): Defines the number of neurons in different hidden layers of the common/
                                        shared layers
            policy_network_size (int): Number of neurons in the hidden layer specific to the policy_network
            value_network_size (int): Number of neurons in the hidden layer specific to the value_network

    """
    #def __init__(self, n_actions, common_network_size=[128,64], policy_network_size=32, value_network_size=32):
    def __init__(self, n_actions, common_network_size=[128,258,512], policy_network_size=[256,128,64], value_network_size=[256,128,64]):
        super(ActorCriticModel_Dimond, self).__init__()
        logger.info("Defining tf model with layers configuration as: {}, {}, {}".format(common_network_size,
                                                                            policy_network_size, value_network_size))
        self.action_size = n_actions
        self.common_hidden_1 = layers.Dense(common_network_size[0], activation='relu')
        self.common_hidden_2 = layers.Dense(common_network_size[1], activation='relu')
        self.common_hidden_3 = layers.Dense(common_network_size[2], activation='relu')
        self.policy_hidden1 = layers.Dense(policy_network_size[0], activation='relu')
        self.policy_hidden2 = layers.Dense(policy_network_size[1], activation='relu')
        self.policy_hidden3 = layers.Dense(policy_network_size[2], activation='relu')
        self.values_hidden1 = layers.Dense(value_network_size[0], activation='relu')
        self.values_hidden2 = layers.Dense(value_network_size[1], activation='relu')
        self.values_hidden3 = layers.Dense(value_network_size[2], activation='relu')
        self.policy_logits = layers.Dense(n_actions)
        self.values = layers.Dense(1)

    def call(self, inputs):
        """Forward pass for the actorcritic model

            The call function is the wrapper on Python's __call__ magic function that is called when a class object is
            directly called without a specific method name.

            The Sub-Classing use this method to implement the forward pass logic.

        """
        # Forward pass
        common = self.common_hidden_1(inputs)
        common = self.common_hidden_2(common)
        common = self.common_hidden_3(common)
        policy_network = self.policy_hidden1(common)
        policy_network = self.policy_hidden2(policy_network)
        policy_network = self.policy_hidden3(policy_network)
        logits = self.policy_logits(policy_network)
        value_network = self.values_hidden1(common)
        value_network = self.values_hidden2(value_network)
        value_network = self.values_hidden3(value_network)
        values = self.values(value_network)
        return logits, values

    
class ActorCriticModel_DoubleDimond(keras.Model):
    """A3C Model

        This class is for the policy and value approximator model for the A3C
        Both the master and the all the workers use individual instances of the same model class.

        This variant of the model class extends the keras.model and use tf.Model Sub-Classing feature.

        In a sub-class(ed) model, advanced features like shared network, shared inputs and residual networks could
        be easily implemented. The network layers need to defined in the __init__() method, and then their connection
        as required in the forward pass needs to be defined in the call or __call__ method.

        TensorFlow eager execution needs to be enabled for this to work as desired.


        Arguments:
            n_action (int): Cardinality of the action_space
            common_network_size (list): Defines the number of neurons in different hidden layers of the common/
                                        shared layers
            policy_network_size (int): Number of neurons in the hidden layer specific to the policy_network
            value_network_size (int): Number of neurons in the hidden layer specific to the value_network
            
            # ST - Original - def __init__(self, n_actions, common_network_size=[128,64], policy_network_size=32, value_network_size=32):
            #  def __init__(self, n_actions, common_network_size=[64,128,256,128,64], policy_network_size=[128,64], value_network_size=[128,64]):
    """
    def __init__(self, n_actions, common_network_size=[128,256,128], policy_network_size=[64,128,64], value_network_size=[64,128,64]):
        super(ActorCriticModel_DoubleDimond, self).__init__()
        logger.info("Defining tf model with layers configuration as: {}, {}, {}".format(common_network_size,
                                                                            policy_network_size, value_network_size))

        self.action_size = n_actions

        self.common_hidden_1 = layers.Dense(common_network_size[0], activation='relu',name="1")
        self.common_hidden_2 = layers.Dense(common_network_size[1], activation='relu',name="2")
        self.common_hidden_3 = layers.Dense(common_network_size[2], activation='relu',name="3")
        self.values_hidden1 = layers.Dense(value_network_size[0], activation='relu',name="7")
        self.policy_hidden1 = layers.Dense(policy_network_size[0], activation='relu',name="4")
        self.values_hidden2 = layers.Dense(value_network_size[1], activation='relu',name="8")
        self.policy_hidden2 = layers.Dense(policy_network_size[1], activation='relu',name="5")
        self.values_hidden3 = layers.Dense(value_network_size[2], activation='relu',name="9")
        self.policy_hidden3 = layers.Dense(policy_network_size[2], activation='relu',name="6")
        self.values = layers.Dense(1,name="11")
        self.policy_logits = layers.Dense(n_actions,name="10")

    def call(self, inputs):
        """Forward pass for the actorcritic model

            The call function is the wrapper on Python's __call__ magic function that is called when a class object is
            directly called without a specific method name.

            The Sub-Classing use this method to implement the forward pass logic.

        """
        
        # Forward pass
        common = self.common_hidden_1(inputs)
        common = self.common_hidden_2(common)
        common = self.common_hidden_3(common)
        policy_network = self.policy_hidden1(common)
        policy_network = self.policy_hidden2(policy_network)
        policy_network = self.policy_hidden3(policy_network)
        value_network = self.values_hidden1(common)
        value_network = self.values_hidden2(value_network)
        value_network = self.values_hidden3(value_network)
        values = self.values(value_network)
        logits = self.policy_logits(policy_network)

        return logits, values
    

if __name__ == "__main__":
    raise NotImplementedError("This class needs to be imported and instantiated from an A3C master/ worker "
                              "agent class and does not contain any invokable code in the main function")