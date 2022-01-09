# -*- coding: utf-8 -*-

'''
@author: zhi zhang

Network agent from agent.py

'''


import numpy as np
import random
import os


from agent import Agent, State
import tensorflow as tf 



class Network():

    
    def __init__(self, exp_config, para_set):

        self.para_set = para_set
        self.exp_config = exp_config
        self.sub_prob = exp_config.SUBPROB 
        self.include_others = exp_config.INCLUDE_OTHERS
        self.include_last_actions = exp_config.INCLUDE_LAST_ACTIONS
        self.include_labels = exp_config.INCLUDE_LABELS  
        self.state = None
        self.action = None
        self.memory = []
        self.average_reward = None

    
    def forget(self):

        ''' remove the old history if the memory is too large '''

        if len(self.memory) > self.para_set.MAX_MEMORY_LEN:
            print("length of memory: {0}, before forget".format(len(self.memory)))
            self.memory = self.memory[-self.para_set.MAX_MEMORY_LEN:]
            print("length of memory: {0}, after forget".format(len(self.memory)))


    

    def fc2(self, t_input, n_hidden=64, n_outputs=9, nonlinearity1=tf.nn.relu,
            nonlinearity2=None, scope='fc2'):
        """
        Two layers
        """
        with tf.variable_scope(scope, initializer=tf.initializers.truncated_normal(0, 0.01)):
            h = tf.layers.dense(inputs=t_input, units=n_hidden,
                                activation=nonlinearity1, use_bias=True,
                                name='h')
            
            out = tf.layers.dense(inputs=h, units=n_outputs,
                                activation=nonlinearity2,
                                use_bias=True, name='out')

        return out


    def fc3(self, t_input, n_hidden1=64, n_hidden2=64, n_outputs=9,
            nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='fc3'):
        """
        Two hidden layer, one output layer
        """
        with tf.variable_scope(scope, initializer=tf.initializers.truncated_normal(0, 0.01)):
            h1 = tf.layers.dense(inputs=t_input, units=n_hidden1,
                                activation=nonlinearity1, use_bias=True,
                                name='h1')

            h2 = tf.layers.dense(inputs=h1, units=n_hidden2,
                                activation=nonlinearity2, use_bias=True,
                                name='h2')
            
            out = tf.layers.dense(inputs=h2, units=n_outputs,
                                activation=None, use_bias=True,
                                name='out')

        return out

    def qmix_agent_net(self, self_obs, self_obs_others, n_actions=2, n_agents=6):
        """
        self_obs: batch*n_agents, 3
        self_obs_others: batch*n_agents, 2*3 matrix
        """
        # self_obs_others = tf.reshape(self_obs_others, [-1, 6])
        self_all_obs = tf.concat([self_obs, self_obs_others], axis=1)
        h1 = tf.layers.dense(inputs=self_all_obs, units=64, 
                    activation=tf.nn.relu, use_bias=True) #[batch*n_agents, 64]
        out = tf.layers.dense(inputs=h1, units=n_actions,
                              activation=None, use_bias=True,
                              name='out')
        return out 


    
    ###### this is the Qmix algorithm, but not know why not working???????
    #####
    def hyper_layer(self, agent_qs, states, n_agents, embed_dim = 32, nonlinearity1 = tf.nn.relu):
        """
        agent_qs: individual q values [batch*n_agents] 
        states: [batch, state_dim]
        """
        # first layer
        # w_1 = tf.abs(self.super_linear(states, embed_dim*n_agents, scope='hyper_w_1'))
        # w_1 = tf.abs(tf.layers.dense(inputs=states, units = embed_dim * n_agents, 
        #                         activation = nonlinearity1, use_bias= False, name = 'hyper_w_1')) 
        scope = "Hyper_layer"
        with tf.variable_scope(scope):
            k = states.get_shape().as_list()[1:]
            bot = -1/np.sqrt(k)
            up = 1/np.sqrt(k)
            # initializer_states = tf.initializers.random_uniform(bot, up)
            initializer_states = tf.zeros_initializer()

            w1 = tf.abs(tf.contrib.layers.fully_connected(states, embed_dim*n_agents, activation_fn=None, 
                                    biases_initializer=initializer_states))#[batch, embed*nagents]
            
            
            # b_1 = tf.layers.dense(inputs=states, units = embed_dim, 
            #                         activation = nonlinearity1, use_bias= False, name = 'hyper_b_1')  #[batch, embed]
            # b_1 = tf.get_variable('hyper_b_1', [embed_dim])
            b1 = tf.contrib.layers.fully_connected(states, embed_dim, activation_fn=None, 
                                    biases_initializer=initializer_states)
            w1 = tf.reshape(w1, [-1, n_agents, embed_dim]) #[batch, n_agents, embed]

            b1 = tf.reshape(b1, [-1, 1, embed_dim])#[batch, 1, embed]

            agent_qs = tf.reshape(agent_qs, [-1, 1, n_agents]) #[batch, 1, n_agents]
                    
            # hidden = tf.nn.relu(tf.nn.bias_add(tf.matmul(agent_qs, w_1), b_1, name='hyper_hidden'))
            hidden = tf.nn.elu(tf.matmul(agent_qs, w1) + b1)  #[batch, 1, embed] expentional linear units 
            
            # second layer 
            
            # w_final = tf.abs(tf.layers.dense(inputs=states, units = embed_dim, 
            #             activation = None, use_bias= nonlinearity1, name = 'hyper_w_final')) #[batch, embed]
            
            w_final = tf.abs(tf.contrib.layers.fully_connected(states, embed_dim, activation_fn=None, 
                                    biases_initializer=initializer_states))#[batch, embed*nagents]
            
            w_final = tf.reshape(w_final, [-1, embed_dim, 1])#[batch, embed, 1]
            
            #---state-dependent bias---
            # b_2 = tf.layers.dense(inputs=states, units = embed_dim, 
            #                         activation = tf.nn.relu, use_bias= False)  #[batch, embed]
            # b_2 = tf.layers.dense(inputs=b_2, units = 1, 
            #                         activation = nonlinearity1, use_bias= False, name = 'hyper_b_2')  #[batch, 1]
            
            b2 = tf.contrib.layers.fully_connected(states, embed_dim, activation_fn=None, 
                                    biases_initializer=initializer_states)  #[batch, embed]
            b2 = tf.nn.relu(b2)
            # b2_initializer = tf.initializers.random_uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim))
            b2_initializer = tf.zeros_initializer()
            b2 = tf.contrib.layers.fully_connected(b2, 1, activation_fn=None, 
                biases_initializer=b2_initializer)  #[batch, 1]
            
            b2 = tf.reshape(b2, [-1,1,1])

            # b_2 = tf.get_variable('hyper_b_2', [1])
            # compute the final output
            y = tf.matmul(hidden, w_final)+b2#[batch, 1, 1] 
            # y = tf.nn.bias_add(tf.matmul(hidden, w_final), b_2, name='hyper_hidden') #[batch, 1, 1]
            # reshape and return 
            q_tot = tf.reshape(y, [-1])
        return q_tot

        
    # def hyper_layer(self, agent_qs, states, n_agents, embed_dim = 32, nonlinearity1 = tf.nn.relu):
    #     """
    #     agent_qs: individual q values [batch*n_agents] 
    #     states: [batch, state_dim]
    #     """
    #     # first layer
    #     scope = "Hyper_layer"
    #     with tf.variable_scope(scope):
    #         agent_qs = tf.reshape(agent_qs, [-1, n_agents])
    #         branch1 = tf.layers.dense(inputs=agent_qs, units=32, activation=tf.nn.relu, use_bias=True, name='hyper_w1')
    #         # to reshape the 2*3 matrix into 6 vector
    #         # branch1 = agent_qs
    #         branch2 = tf.layers.dense(inputs=states, units=32, activation=tf.nn.relu, use_bias=True, name='hyper_w2')
    #         concated = tf.concat([branch1, branch2], axis=1)
    #         W_concated_out = self.get_variable("W_concated_out", [32*2, 32])
    #         l1_out = tf.matmul(concated, W_concated_out) 

    #         h2 = tf.abs(tf.layers.dense(inputs=l1_out, units=32,
    #                             activation=tf.nn.relu, use_bias=True,
    #                             name='h2'))
            
    #         out = tf.layers.dense(inputs=h2, units=1,
    #                             activation=None, use_bias=True,
    #                             name='out')
            
    #         out = tf.reshape(out, [-1])
    #     return out 



    def convnet(self, t_input, f1=4, k1=[10,5], s1=[5,2],
                f2=8, k2=[6,3], s2=[3,2], scope='convnet'):
        """
        f1 - number of filters in first layer
        k1 - kernel size of first layer
        s1 - stride of first layer
        f2 - number of filters in second layer
        k2 - kernel size of second layer
        s2 - stride of second layer
        """
        if len(t_input.shape) != 4:
            raise ValueError("networks.py convnet : t_input must be 4D tensor")
        with tf.variable_scope(scope):
            conv1 = tf.contrib.layers.conv2d(inputs=t_input, num_outputs=f1,
                                            kernel_size=k1, stride=s1,
                                            padding="SAME",
                                            activation_fn=tf.nn.relu)
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=f2,
                                            kernel_size=k2, stride=s2,
                                            padding="SAME",
                                            activation_fn=tf.nn.relu)
            # ignore the first dimension, corresponding to batch size
            size = np.prod(conv2.get_shape().as_list()[1:])
            conv2_flat = tf.reshape(conv2, [-1, size])

        return conv2_flat
    

    def get_variable(self, name, shape):

        return tf.get_variable(name, shape, tf.float32,
                            tf.initializers.truncated_normal(0,0.01))

    def convnet_1(self, t_input, f1=4, k1=[5,3], s1=[1,1], scope='convnet_1'):
        if len(t_input.shape) != 4:
            raise ValueError("networks.py convnet_1 : t_input must be 4D tensor")
        with tf.variable_scope(scope):
            conv1 = tf.contrib.layers.conv2d(inputs=t_input, num_outputs=f1, kernel_size=k1, stride=s1, padding="SAME", activation_fn=tf.nn.relu)
            size = np.prod(conv1.get_shape().as_list()[1:])
            conv1_flat = tf.reshape(conv1, [-1, size])

        return conv1_flat

    def Q_rnn(self, self_obs, self_obs_others, last_actions, agent_labels, sub_labels, n_steps, prev_state, n_actions=2, n_agents=6):

        """
        self_obs: batch*n_agents, 3*n_agents
        self_obs_others: batch*n_agents,  6*n_agents
        """
        
        gru = tf.nn.rnn_cell.GRUCell(num_units=64,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.initializers.truncated_normal(0,0.01),
                                bias_initializer=tf.initializers.random_uniform(-1/np.sqrt(64), 1/np.sqrt(64)))

        # Concatenate with vectors
        # self_obs_others = tf.reshape(self_obs_others, [-1, 6], name='rnn_obs_one_agent')
        # branch1 = tf.layers.dense(inputs=self_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=True, name='v_local_branch2')
        if self.sub_prob == True:
            agent_labels = sub_labels

        # if self.include_others and self.include_last_actions:
        #     concated = tf.concat([self_obs, self_obs_others, last_actions, agent_labels], 1)
        # if not self.include_others and self.include_last_actions:
        #     concated = tf.concat([self_obs, last_actions, agent_labels], 1)
        # if self.include_others and not self.include_last_actions:
        #     concated = tf.concat([self_obs, self_obs_others, agent_labels], 1)
        # if not self.include_others and not self.include_last_actions:
        #     concated = tf.concat([self_obs, agent_labels], 1)
        flags = [self.include_others, self.include_last_actions, self.include_labels]
        items = [self_obs_others, last_actions, agent_labels]
        concated = self_obs
        for flag, item in zip(flags, items):
            if flag:
                concated = tf.concat([concated, item], 1) 
        



        k = concated.get_shape().as_list()[1:]
        bot = -1/np.sqrt(k)
        up = 1/np.sqrt(k)
        initializer_states = tf.initializers.random_uniform(bot, up)
        # tf.zeros_initializer()
        x = tf.nn.relu(tf.contrib.layers.fully_connected(concated, 64, activation_fn=None, 
                                biases_initializer=initializer_states))#[batch*agents,hiddenunits]
        

        # Reshape to [n_steps, n_agents, ...] 
        # then switch to [n_agents, n_steps, ...] for RNN processing
        last_dim_size = x.get_shape().as_list()[-1]
        rnn_in = tf.transpose(tf.reshape(x, [n_steps, n_agents, last_dim_size]), [1,0,2])
        
        # Process with GRU layer
        # gru_out shape is [n_agents, n_steps, n_units]
        # state shape is [n_agents, n_units] since the hidden state is shared and updated by all time steps 
        gru_out, state = tf.nn.dynamic_rnn(gru, rnn_in,
                                    initial_state=prev_state,
                                    dtype=tf.float32)

        # Switch back to [n_steps, n_agents, n_units], then back to
        # [n_steps*n_agents, n_units], for compatibility with alg.py
        rnn_out = tf.reshape(tf.transpose(gru_out, [1,0,2]), [-1, 64])

        # One fc layer
        # q = self.fc2(rnn_out, 64, n_actions, tf.nn.relu, scope='rnn_fc') #[batch*gents, n_actions]
        q = tf.contrib.layers.fully_connected(rnn_out, n_actions, activation_fn=None, 
            biases_initializer=tf.initializers.random_uniform(-1/np.sqrt(64), 1/np.sqrt(64)))
        return q, state

    
    def V_i(self, self_obs, self_obs_others, last_actions, agent_labels):
        """
        Used by IAC 
        """
        n_h1 = 32
        n_hidden2 = 64
        units = 256

        # branch1 = tf.layers.dense(inputs=self_obs, units=n_h1, activation=tf.nn.relu, use_bias=True, name='v_local_branch1')
        # branch2 = tf.layers.dense(inputs=self_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=True, name='v_local_branch2')
        # concated = tf.concat( [branch1, self_last_actions, self_agent_labels], axis=1)
        # # # # W_concated_out = self.get_variable("W_concated_out", [n_h1, n_h1])
        # # # # l1_out = tf.matmul(concated, W_concated_out) 
        # # # l1_out = concated
        # l1_out = tf.concat([self_obs, self_last_actions, self_agent_labels], axis=1)
        # h2 = tf.layers.dense(inputs=l1_out, units=n_hidden2,
        #                      activation=tf.nn.relu, use_bias=True,
        #                      name='h2')
        # out = tf.layers.dense(inputs=h2, units=1,
        #                       activation=None, use_bias=True,
        #                       name='out') #[batch*n_agents]

        # if self.include_others and self.include_last_actions:
        #     concated = tf.concat([self_obs, self_obs_others, last_actions, agent_labels], 1)
        # if not self.include_others and self.include_last_actions:
        #     concated = tf.concat([self_obs, last_actions, agent_labels], 1)
        # if self.include_others and not self.include_last_actions:
        #     concated = tf.concat([self_obs, self_obs_others, agent_labels], 1)
        # if not self.include_others and not self.include_last_actions:
        #     concated = tf.concat([self_obs, agent_labels], 1)
        flags = [self.include_others, self.include_last_actions, self.include_labels]
        items = [self_obs_others, last_actions, agent_labels]
        concated = self_obs
        for flag, item in zip(flags, items):
            if flag:
                concated = tf.concat([concated, item], 1) 
        

        
        with tf.variable_scope("i_v"):
            ## two hidden layers + one output layers 
            out = self.fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=1, 
                  nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='IAC')

        return out

    def Q_i(self, self_obs, self_obs_others,  last_actions, agent_labels, sub_labels, n_actions=2):
        n_h1 = 32
        n_hidden2 = 32
        units = 256
        # branch1 = tf.layers.dense(inputs=self_obs, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_local_branch1')
        # # # branch2 = tf.layers.dense(inputs=self_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_local_branch2')
        # concated = tf.concat( [branch1, last_actions, agent_labels], axis=1)
        # s1 = last_actions.get_shape().as_list()[1] 
        # s2 = agent_labels.get_shape().as_list()[1]
        # W_concated_out = self.get_variable("W_concated_out", [n_h1+s1+s2, n_h1])
        # l1_out = tf.matmul(concated, W_concated_out) 
        # h2 = tf.layers.dense(inputs=l1_out, units=n_hidden2,
        #                       activation=tf.nn.relu, use_bias=True,
        #                       name='h2')
        # out = tf.layers.dense(inputs=h2, units=n_actions,
        #                        activation=None, use_bias=True,
        #                        name='out') #[batch, n_agents*n_actions]
        if self.sub_prob == True:
            agent_labels = sub_labels
        # if self.include_others and self.include_last_actions:
        #     concated = tf.concat([self_obs, self_obs_others, last_actions, agent_labels], 1)
        # if not self.include_others and self.include_last_actions:
        #     concated = tf.concat([self_obs, last_actions, agent_labels], 1)
        # if self.include_others and not self.include_last_actions:
        #     concated = tf.concat([self_obs, self_obs_others, agent_labels], 1)
        # if not self.include_others and not self.include_last_actions:
        #     concated = tf.concat([self_obs, agent_labels], 1)

        flags = [self.include_others, self.include_last_actions, self.include_labels]
        items = [self_obs_others, last_actions, agent_labels]
        concated = self_obs
        for flag, item in zip(flags, items):
            if flag:
                concated = tf.concat([concated, item], 1) 
        

        
        with tf.variable_scope("i_Q"):
            ## two hidden layers + one output layers 
            out = self.fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=n_actions, 
                  nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='Individual_Q')
        
        return out 

    def mix_i(self, self_obs, self_obs_others, last_actions, agent_labels, state, sub_labels, n_actions=2):
        n_h1 = 32
        n_hidden2 = 64
        units = 256
        # branch1 = tf.layers.dense(inputs=state, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_local_branch1')
        # branch2 = tf.layers.dense(inputs=self_obs, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_local_branch2')
        # branch3 = tf.layers.dense(inputs=self_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_local_branch3')
        # concated = tf.concat( [branch2, branch3], axis=1)
        # W_concated_out = self.get_variable("W_concated_out", [2*n_h1, n_h1])
        # l1_out = tf.matmul(concated, W_concated_out) 
        # concated = tf.concat( [branch2, last_actions, agent_labels], axis=1)
        # l1_out = tf.layers.dense(inputs=concated, units=n_h1, activation=tf.nn.relu, use_bias=True, name='Q_local_l1_out')
        # concated = tf.concat([state, actions_reshaped, agent_labels, self_obs, self_obs_others], axis=1)
        if self.sub_prob == True:
            agent_labels = sub_labels
        
        flags = [self.include_others, self.include_last_actions, self.include_labels]
        items = [self_obs_others, last_actions, agent_labels]
        concated = self_obs
        for flag, item in zip(flags, items):
            if flag:
                concated = tf.concat([concated, item], 1) 
        
        with tf.variable_scope("mix_q"):
            ## two hidden layers + one output layers 
            out = self.fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=n_actions, 
                  nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='QCombo_i')
         #[batch, n_agents*n_actions]
        return out 


    def Q_g(self, t_global, actions, n_actions=2, n_agents=6, units=256):
        # n_agents = actions.get_shape().as_list()[1] 
        #flatten out the joint actions, the original actions is [time*n_agents, n_actions]
        actions_reshaped = tf.reshape(actions, [-1, n_agents*n_actions])
        #concatenate with all other features
        concated = tf.concat([t_global, actions_reshaped], axis=1)
        with tf.variable_scope('global_Q'):
            out = self.fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=1, 
                nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='QCombo_g')
        return out 


    def Q_g_rnn(self, t_global, actions, n_steps, prev_state, n_actions=1, n_agents=1):

        """
        t_global: batch*n_agents, 3*n_agents
        actions: batch*n_agents,  6*n_agents
        """
        
        gru = tf.nn.rnn_cell.GRUCell(num_units=64,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.initializers.truncated_normal(0,0.01),
                                bias_initializer=tf.initializers.random_uniform(-1/np.sqrt(64), 1/np.sqrt(64)))

        # Concatenate with vectors
        # self_obs_others = tf.reshape(self_obs_others, [-1, 6], name='rnn_obs_one_agent')
        # branch1 = tf.layers.dense(inputs=self_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=True, name='v_local_branch2')
        actions_reshaped = tf.reshape(actions, [-1, n_agents*n_actions])
        concated = tf.concat([t_global, actions_reshaped], axis=1) # [nsteps, dimensions] global q doesn't have n agents 
        
        k = concated.get_shape().as_list()[1:]
        bot = -1/np.sqrt(k)
        up = 1/np.sqrt(k)
        initializer_states = tf.initializers.random_uniform(bot, up)
        # tf.zeros_initializer()
        x = tf.nn.relu(tf.contrib.layers.fully_connected(concated, 64, activation_fn=None, 
                                biases_initializer=initializer_states))#[batch,hiddenunits]
        

        # Reshape to [n_steps, n_agents, ...] 
        # then switch to [n_agents, n_steps, ...] for RNN processing
        last_dim_size = x.get_shape().as_list()[-1]
        rnn_in = tf.transpose(tf.reshape(x, [n_steps, n_agents, last_dim_size]), [1,0,2])
        
        # Process with GRU layer
        # gru_out shape is [n_agents, n_steps, n_units]
        # state shape is [n_agents, n_units] since the hidden state is shared and updated by all time steps 
        gru_out, state = tf.nn.dynamic_rnn(gru, rnn_in,
                                    initial_state=prev_state,
                                    dtype=tf.float32)

        # Switch back to [n_steps, n_agents, n_units], then back to
        # [n_steps*n_agents, n_units], for compatibility with alg.py
        rnn_out = tf.reshape(tf.transpose(gru_out, [1,0,2]), [-1, 64])

        # One fc layer
        # q = self.fc2(rnn_out, 64, n_actions, tf.nn.relu, scope='rnn_fc') #[batch*gents, n_actions]
        q = tf.contrib.layers.fully_connected(rnn_out, 1, activation_fn=None, 
            biases_initializer=tf.initializers.random_uniform(-1/np.sqrt(64), 1/np.sqrt(64)))
        return q, state




    def Q_coma(self, self_obs, self_obs_others, action_others, agent_labels, state, n_actions=2, units=256):
        """
        Used for testing pure COMA on environment
        """
        # self_obs_others = tf.reshape(self_obs_others, [-1, 6], name='obs_one_agent')
        n_others = action_others.get_shape().as_list()[1]
        actions_reshaped = tf.reshape(action_others, [-1, n_others*n_actions])
        if self.include_others:
            concated = tf.concat([state, actions_reshaped, agent_labels, self_obs, self_obs_others], axis=1)
        else:
            concated = tf.concat([state, actions_reshaped, agent_labels, self_obs], axis=1)


        with tf.variable_scope("coma_q"):
            ## two hidden layers + one output layers 
            out = self.fc3(concated, n_hidden1=units, n_hidden2=units, n_outputs=n_actions, 
                  nonlinearity1=tf.nn.relu, nonlinearity2=tf.nn.relu, scope='Q_coma')
        return out


    def actor_staged(self, self_obs, self_obs_others,  
                 n_h1=32, n_h2=64, n_actions=9, stage=1):
        """
        """
        branch1 = tf.layers.dense(inputs=self_obs, units=n_h1,
                                activation=tf.nn.relu, use_bias=True,
                                name='actor_branch1')

        # self_obs_others = tf.reshape(self_obs_others, [-1, 6], name='obs_one_agent')

        if self.include_others:
            branch2 = tf.layers.dense(inputs=self_obs_others, units=n_h1,
                                    activation=tf.nn.relu, use_bias=True,
                                    name='actor_branch2')

            # stack the two branches together
            concated = tf.concat([branch1, branch2], 1)
            # craete new weights 
            W_concated_h2 = self.get_variable("W_concated_h2", [2*n_h1, n_h2])
        else:
            concated = branch1 
            W_concated_h2 = self.get_variable("W_concated_h2", [n_h1, n_h2])
        
        list_mult = []
        list_mult.append( tf.matmul(concated, W_concated_h2) )

        b = tf.get_variable('b', [n_h2])
        h2 = tf.nn.relu(tf.nn.bias_add(tf.add_n(list_mult), b))

        # Output layer
        out = tf.layers.dense(inputs=h2, units=n_actions, activation=None, use_bias=True, name='actor_out')
        

        probs = tf.nn.softmax(out, name='actor_softmax') #[batch*n_agents, n_actions]

        return probs
        
    def actor_staged_rnn(self, self_obs, self_obs_others, n_steps, prev_state, n_actions, n_agents, n_h1=32, n_h2=64):
        """
        """

        gru = tf.nn.rnn_cell.GRUCell(num_units=64,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.initializers.truncated_normal(0,0.01),
                                bias_initializer=tf.initializers.random_uniform(-1/np.sqrt(64), 1/np.sqrt(64)))

        # Concatenate with vectors
        # self_obs_others = tf.reshape(self_obs_others, [-1, 6], name='rnn_obs_one_agent')
        # branch1 = tf.layers.dense(inputs=self_obs_others, units=n_h1, activation=tf.nn.relu, use_bias=True, name='v_local_branch2')
        
        if self.include_others:
            concated = tf.concat([self_obs, self_obs_others], 1)
        else:
            concated = self_obs

        k = concated.get_shape().as_list()[1:]
        bot = -1/np.sqrt(k)
        up = 1/np.sqrt(k)
        initializer_states = tf.initializers.random_uniform(bot, up)
        # tf.zeros_initializer()
        x = tf.nn.relu(tf.contrib.layers.fully_connected(concated, 64, activation_fn=None, 
                                biases_initializer=initializer_states))#[batch*agents,hiddenunits]
        

        # Reshape to [n_steps, n_agents, ...] 
        # then switch to [n_agents, n_steps, ...] for RNN processing
        last_dim_size = x.get_shape().as_list()[-1]
        rnn_in = tf.transpose(tf.reshape(x, [n_steps, n_agents, last_dim_size]), [1,0,2])
        
        # Process with GRU layer
        # gru_out shape is [n_agents, n_steps, n_units]
        # state shape is [n_agents, n_units] since the hidden state is shared and updated by all time steps 
        gru_out, state = tf.nn.dynamic_rnn(gru, rnn_in,
                                    initial_state=prev_state,
                                    dtype=tf.float32)

        # Switch back to [n_steps, n_agents, n_units], then back to
        # [n_steps*n_agents, n_units], for compatibility with alg.py
        rnn_out = tf.reshape(tf.transpose(gru_out, [1,0,2]), [-1, 64])

        # One fc layer
        # q = self.fc2(rnn_out, 64, n_actions, tf.nn.relu, scope='rnn_fc') #[batch*gents, n_actions]
        q = tf.contrib.layers.fully_connected(rnn_out, n_actions, activation_fn=None, 
            biases_initializer=tf.initializers.random_uniform(-1/np.sqrt(64), 1/np.sqrt(64)))

        
        
        probs = tf.nn.softmax(q, name='actor_softmax') #[batch*n_agents, n_actions]


        
        return probs, state

        

    def super_linear(self, x, output_size, scope=None, reuse=False, init_w="ortho", weight_start=0.0, use_bias=True, bias_start=0.0):
        # support function doing linear operation.  uses ortho initializer defined earlier.
        shape = x.get_shape().as_list()
        with tf.variable_scope(scope or "linear"):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()
            w_init = None # uniform
            x_size = shape[1]
            h_size = output_size
            if init_w == "zeros":
                w_init=tf.constant_initializer(0.0)
            elif init_w == "constant":
                w_init=tf.constant_initializer(weight_start)
            elif init_w == "gaussian":
                w_init=tf.random_normal_initializer(stddev=weight_start)
            elif init_w == "ortho":
                w_init=self.orthogonal_initializer(1.0)

            w = tf.get_variable("super_linear_w",
            [shape[1], output_size], tf.float32, initializer=w_init)
            if use_bias:
                b = tf.get_variable("super_linear_b", [output_size], tf.float32,
                initializer=tf.constant_initializer(bias_start))
                return tf.matmul(x, w) + b
            return tf.matmul(x, w)
    
    def orthogonal_initializer(self, scale=1.0):
        def _initializer(shape, dtype=tf.float32, partition_info=None):
            return tf.constant(self.orthogonal(shape) * scale, dtype)
        return _initializer

    def orthogonal(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)


   


    

        



    