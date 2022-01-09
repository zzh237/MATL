# -*- coding: utf-8 -*-

'''
@author: zhi zhang

Deep reinforcement learning agent

'''

import numpy as np
import random
import os
from a04network_agent import Network
import tensorflow as tf
import numpy as np
import sys
import json
import shutil



MEMO = "Deeplight"

class Alg():

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

            if hasattr(self, "STATE_FEATURE"):
                self.LIST_STATE_FEATURE = []
                list_state_feature_names = list(self.STATE_FEATURE.keys())
                list_state_feature_names.sort()
                for feature_name in list_state_feature_names:
                    if self.STATE_FEATURE[feature_name]:
                        self.LIST_STATE_FEATURE.append(feature_name)


    def __init__(self, exp_conf, path_set, 
                 dimensions, rank, n_agents=1, num_actions = 2,
                 tau=0.01, lr_V=0.001, lr_Q=0.001,
                 lr_actor=0.0001, gamma=0.99):

        """
        Inputs:
        dimensions - dictionary containing tensor dimensions
                     (h,w,c) for tensor
                     l for 1D vector
        tau - target variable update rate
        lr_V, lr_Q, lr_actor - learning rates for optimizer
        gamma - discount factor
        alpha - weighting of local vs. global gradient
        use_Q - if 1, activates Q network
        use_V - if 1, activates V network
        """

        self.path_set = path_set
        conf_file = os.path.join(self.path_set.PATH_TO_CONF, self.path_set.AGENT_CONF)
        dic_paras = json.load(open(conf_file, "r"))
        #return key value pair to initiate the para_set
        self.para_set = self.ParaSet(dic_paras)
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.AGENT_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.AGENT_CONF))

        self.exp_config = exp_conf
        self.path_set = path_set 
        self.n_actions = num_actions
        self.average_reward = None
        # global self_allsame_obs is all agents' normalized (x,y,speed)
        # agent's own velocity and position
        self.rank = rank 
        self.sub_prob = exp_conf.SUBPROB
    
        self.rnn = exp_conf.RNN
        self.n_agents = n_agents
        self.tau = tau
        self.lr_V = lr_V
        self.lr_Q = lr_Q
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.alpha = self.para_set.ALPHA 
        self.batch = self.para_set.BATCH_SIZE 
        # self.use_Q = use_Q
        # self.use_V = use_V
        # self.nn = nn
        # self.IAC = IAC
        self.agent_labels = np.eye(self.n_agents)
        # self.pv = tf.get_variable("hidden_state", [self.n_agents,64], tf.float32,tf.zeros_initializer())

        self.self_obs_dimension = dimensions['self_obs']
        self.global_state_dimension = dimensions['global_state']

        # Initialize computational graph  
        self.networks = Network(exp_conf, self.para_set)
        self.create_networks()
        if self.para_set.TRAIN_TEST == 'train':
            self.list_initialize_target_ops, self.list_update_target_ops = self.get_assign_target_ops(tf.trainable_variables())
            if self.para_set.MODEL_NAME in ['IDQN']:
                self.create_local_dqn_train_op()
            if self.para_set.MODEL_NAME in ['QCombo']:
                self.create_global_dqn_train_op()
            if self.para_set.MODEL_NAME in ['IAC']:
                self.create_local_critic_train_op()
                self.create_policy_gradient_op()
            if self.para_set.MODEL_NAME in ['QCombo','VDN','QMix']:
                self.create_dqn_policy()
            if self.para_set.MODEL_NAME in ['COMA']:
                self.create_global_critic_train_op()
                self.create_policy_gradient_op()
        # TF summaries
            if self.para_set.MODEL_NAME in ['QCombo']:
                self.create_summary()
    
    def create_networks(self):
        # Placeholders
        # None implicitely means the the times * agents
        # self.self_obs_part1_i = tf.placeholder(tf.float32, [None, 3], 'state_one_agent')
        # self.self_obs_i = tf.placeholder(tf.float32, [None, 6], 'obs_one_agent')
        
        self.self_obs = tf.placeholder(tf.float32, [None, self.self_obs_dimension], 'state_self_view')
        self.self_obs_others = tf.placeholder(tf.float32, [None, (self.n_agents-1)*self.self_obs_dimension], 'other_obs_self_view')
        # self.self_obs = tf.placeholder(tf.float32, [None, 3], 'state_self_view')
        # self.self_obs_others = tf.placeholder(tf.float32, [None, 6], 'obs_self_view')
        

        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        self.self_agent_qs = tf.placeholder(tf.float32, [None], 'self_agent_qs')
        
        ## uniform global self_allsame_obs same for all agents 
        self.global_state = tf.placeholder(tf.float32, [None, self.global_state_dimension], 'global_state')
        
        self.self_global_state = tf.placeholder(tf.float32, [None, self.global_state_dimension], 'self_global_state')
        # self.self_global_state = tf.placeholder(tf.float32, [None, 3*self.n_agents], 'global_state_coma')
        
        
        self.self_actions = tf.placeholder(tf.float32, [None, self.n_actions], 'self_actions')
        self.self_actions_others = tf.placeholder(tf.float32, [None, self.n_agents-1, self.n_actions], 'self_actions_others')
        self.self_last_actions = tf.placeholder(tf.float32, [None, self.n_actions], 'self_last_actions')
        
        self.self_agent_labels = tf.placeholder(tf.float32, [None, self.n_agents])
        self.n_steps = tf.placeholder(tf.float32, None, 'n_steps')
        ##this is due to some issues 
        self.self_sub_labels = tf.placeholder(tf.float32, [None, 4], 'sub_prob_labels')

        ## the prev state is used as initialized hidden state
        self.prev_state = tf.placeholder(tf.float32, [self.n_agents,64], 'self_prev_state')
        self.prev_state_target = tf.placeholder(tf.float32, [self.n_agents,64], 'self_prev_state')   
        
        if self.para_set.MODEL_NAME in ['IDQN', 'QCombo', 'QMix', 'VDN']:
            # Individual DQN network  
            # The output variables here are just structure, you need to do more work in order to do the optimization
            with tf.variable_scope('Q_i_main'):
                # self_allsame_obs shape is [n_agents, n_units]
                # self.prev_state = tf.get_variable("hidden_state", [self.n_agents,64], tf.float32,tf.zeros_initializer())
                #  tf.initializers.truncated_normal(0,0.01))  
                if self.rnn: 
                    self.Q_i_main, self.out_state = self.networks.Q_rnn(self.self_obs, 
                                self.self_obs_others, self.self_last_actions, 
                                self.self_agent_labels, self.self_sub_labels,   
                                self.n_steps, self.prev_state, self.n_actions, self.n_agents)

                else: 
                    self.Q_i_main = self.networks.Q_i(self.self_obs, self.self_obs_others, 
                            self.self_last_actions, 
                            self.self_agent_labels, self.self_sub_labels, n_actions=self.n_actions)
                #self.Q_i_main: [time*n_agents, n_actions], action_samples: [time*n_agents]
            #greedy select actions from Q individual                            
            self.action_samples = tf.argmax(self.Q_i_main,axis=1)  
            
            with tf.variable_scope('Q_i_target'):
                if self.rnn: 
                    self.Q_i_target, self.out_state_target = self.networks.Q_rnn(self.self_obs, 
                            self.self_obs_others, self.self_last_actions, 
                            self.self_agent_labels, self.self_sub_labels, 
                            self.n_steps, self.prev_state_target, self.n_actions, self.n_agents)
                else: 
                    self.Q_i_target = self.networks.Q_i(self.self_obs, self.self_obs_others, 
                            self.self_last_actions, self.self_agent_labels, self.self_sub_labels, 
                            n_actions=self.n_actions)
            self.action_samples_target = tf.argmax(self.Q_i_target,axis=1)
                
       
        if self.para_set.MODEL_NAME in ['QCombo']:                              
            if self.sub_prob == False: 
                with tf.variable_scope('qcombo_g_main'):
                    #self.self_actions [time*n_agents, 1_hot_actions_vector]
                    self.qcombo_g_main = self.networks.Q_g(self.global_state, self.self_actions, 
                                            n_actions=self.n_actions, n_agents=self.n_agents)
                with tf.variable_scope('qcombo_g_target'):
                    self.qcombo_g_target = self.networks.Q_g(self.global_state, self.self_actions, 
                                        n_actions=self.n_actions, n_agents=self.n_agents)
        
        if self.para_set.MODEL_NAME in ['QCombo', 'VDN','QMix']:
            with tf.variable_scope('Q_agents_main'): 
                self.Q_mix_agents_main = tf.reduce_sum(tf.multiply(self.Q_i_main, 
                self.self_actions), axis=1)
                #[batch*n_agents] 
            # with tf.variable_scope('Q_agents_target'): 
            #     self.Q_mix_agents_target = tf.reduce_sum(tf.multiply(self.Q_i_target, 
            #     self.self_actions), axis=1)
        if self.para_set.MODEL_NAME in ['QMix']:    
            with tf.variable_scope('QMix_main'):    
                self.QMix_g_main = self.networks.hyper_layer(self.Q_mix_agents_main, self.global_state,  
                            n_agents = self.n_agents, embed_dim = 128, nonlinearity1 = tf.nn.relu)
            
            with tf.variable_scope('QMix_target'):    
                #please feed in the self_agent_qs
                self.QMix_g_target = self.networks.hyper_layer(self.self_agent_qs, self.global_state,  
                            n_agents = self.n_agents, embed_dim = 128, nonlinearity1 = tf.nn.relu)
        
        if self.para_set.MODEL_NAME in ['VDN']:
            with tf.variable_scope('VDN_main'):
                #[batch*n_agents]
                self.vdn_g_main = tf.reshape(self.Q_mix_agents_main, [-1, self.n_agents]) #[batch, n_agents]
                self.vdn_g_main = tf.reduce_sum(self.vdn_g_main, axis=1) #[batch]
            with tf.variable_scope('VDN_target'):
                ##please feed in the self_agent_qs #[batch*n_agents]
                self.vdn_g_target = tf.reshape(self.self_agent_qs, [-1, self.n_agents]) #[batch, n_agents]
                self.vdn_g_target = tf.reduce_sum(self.vdn_g_target, axis=1) #[batch]

        if self.para_set.MODEL_NAME in ['IAC']:
            with tf.variable_scope("V_i_main"):
                self.V_i_main = self.networks.V_i(self.self_obs, self.self_obs_others, 
                self.self_last_actions, self.self_agent_labels)
            with tf.variable_scope("V_i_target"):
                self.V_i_target = self.networks.V_i(self.self_obs, self.self_obs_others,
                self.self_last_actions, self.self_agent_labels)


        # Q(s, a^{-n}, g^n, g^{-n}, n, o^n)
        if self.para_set.MODEL_NAME in ['COMA']:
            with tf.variable_scope("Q_coma_main"):
                self.Q_coma = self.networks.Q_coma(self.self_obs, self.self_obs_others, 
                            self.self_actions_others, self.self_agent_labels, self.self_global_state, 
                            n_actions=self.n_actions, units=self.para_set.Q_units) #[batch*n_agents, actions]
                
            
            with tf.variable_scope("Q_coma_target"):
                self.Q_coma_target = self.networks.Q_coma(self.self_obs, self.self_obs_others, 
                            self.self_actions_others, self.self_agent_labels, self.self_global_state, 
                            n_actions=self.n_actions, units=self.para_set.Q_units)
        
        if self.para_set.MODEL_NAME in ['IAC','COMA']:
            with tf.variable_scope("Policy_main"):
                if self.rnn:
                    probs, self.out_state= self.networks.actor_staged_rnn(self.self_obs, self.self_obs_others, 
                    self.n_steps, self.prev_state, self.n_actions, self.n_agents)
                        
                else:
                    probs = self.networks.actor_staged(self.self_obs, self.self_obs_others, 
                        n_actions=self.n_actions)
            
            # probs is normalized
            self.probs = (1-self.epsilon) * probs + self.epsilon/float(self.n_actions)
            self.action_samples_actor = tf.multinomial(tf.log(self.probs), 1)
            with tf.variable_scope("Policy_target"):
                if self.rnn:
                    probs_target, self.out_state_target= self.networks.actor_staged_rnn(self.self_obs, self.self_obs_others, 
                        self.n_steps, self.prev_state_target, self.n_actions, self.n_agents)
                else:
                    probs_target = self.networks.actor_staged(self.self_obs, self.self_obs_others, 
                    n_actions=self.n_actions)
        
            
            self.action_samples_target_actor = tf.multinomial(tf.log( (1-self.epsilon)*probs_target 
                                    + self.epsilon/float(self.n_actions) ), 1)


    def create_local_dqn_train_op(self):
        # self.self_actions = tf.placeholder(tf.float32, [None, self.n_actions], 'actions_self_1hot')
        self.Q_target = tf.placeholder(tf.float32, [None], 'Q_target')
        self.Q_taken = tf.reduce_sum(tf.multiply(self.Q_i_main, self.self_actions), axis=1)
        self.loss_Q = tf.reduce_mean(tf.square(self.Q_target - self.Q_taken))
        self.Q_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_op = self.Q_opt.minimize(self.loss_Q)

    def create_global_dqn_train_op(self):
        self.Q_target = tf.placeholder(tf.float32, [None], 'Q_target')
        self.Q_g_bellman = tf.placeholder(tf.float32, [None], 'Q_g_bellman')
        self.loss_g_Q = tf.reduce_mean(tf.square(self.Q_g_bellman - self.qcombo_g_main))   
        #using the RNN Q instead of the individual Q
    

        #using the COMA Q instead of the individual Q
        # self.Q_coma_target = tf.placeholder(tf.float32, [None], 'Q_target')
        # self.Q_coma_taken= tf.reduce_sum(tf.multiply( self.Q_coma, self.self_actions ), axis=1 )  
        # self.loss_i_Q = tf.reduce_mean(tf.square(tf.reduce_sum(tf.reshape(self.Q_target - self.Q_taken, [-1, self.n_agents]), axis=1))) 
        # self.overall_loss = self.alpha * self.loss_i_Q + (1-self.alpha) * self.loss_g_Q
        # self.overall_loss = self.loss_g_Q
        # self.Q_g_opt = tf.train.AdamOptimizer(self.lr_Q)
        # self.Q_g_op = self.Q_g_opt.minimize(self.loss_g_Q)
    def create_local_critic_train_op(self):
        # TD target calculated in train_step() using V_target
        self.V_td_target = tf.placeholder(tf.float32, [None], 'V_td_target') #[batch*n_agents]
        self.loss_V = tf.reduce_mean(tf.square(self.V_td_target - tf.squeeze(self.V_i_main)))
        self.V_opt = tf.train.AdamOptimizer(self.lr_V)
        self.V_op = self.V_opt.minimize(self.loss_V)

    
    def create_global_critic_train_op(self):
        # TD target calculated in train_step() using Q_target
        self.Q_td_target = tf.placeholder(tf.float32, [None], 'Q_td_target')
        # self.self_actions = tf.placeholder(tf.float32, [None, self.n_actions], 'actions_self_1hot')
        # Get Q-value of action actually taken by point-wise mult
        self.Q_action_taken = tf.reduce_sum(tf.multiply( self.Q_coma, self.self_actions ), axis=1 )
        self.gc_loss_Q = tf.reduce_mean(tf.square(self.Q_td_target - self.Q_action_taken)) #gc means global_critic
        self.gc_Q_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.gc_Q_op = self.gc_Q_opt.minimize(self.gc_loss_Q)



    def create_policy_gradient_op(self):

        # batch of 1-hot action vectors
        # self.self_actions = tf.placeholder(tf.float32, [None, self.n_actions], 'action_taken')
        # self.probs has shape [batch, l_action]
        log_probs = tf.log(tf.reduce_sum(tf.multiply(self.probs, self.self_actions), axis=1) + 1e-15) #[batch*nagents]

        # --------------- COMA -------------------- #
        if self.para_set.MODEL_NAME in ['COMA']:
            # Q-values of the action actually taken [batch_size]
            self.Q_evaluated = tf.placeholder(tf.float32, [None, self.n_actions], 'Q_evaluated')
            self.COMA_1 = tf.reduce_sum(tf.multiply(self.Q_evaluated, self.self_actions), axis=1)
            # Use all Q-values at output layer [batch_size]
            self.probs_evaluated = tf.placeholder(tf.float32, [None, self.n_actions])
            self.COMA_2 = tf.reduce_sum(tf.multiply(self.Q_evaluated, self.probs_evaluated), axis=1)
            self.COMA = tf.subtract(self.COMA_1, self.COMA_2)
            # self.policy_loss_global = -tf.reduce_mean( tf.multiply( log_probs, self.COMA ) )
            self.policy_loss_global = -tf.reduce_mean( tf.reduce_sum( tf.reshape( tf.multiply(log_probs, self.COMA), 
                                        [-1, self.n_agents] ), axis=1) )
            # -------------- End COMA ----------------- #
            self.policy_loss = self.policy_loss_global
        
        if self.para_set.MODEL_NAME in ['IAC']:
            self.V_evaluated = tf.placeholder(tf.float32, [None], 'V_evaluated')
            self.V_td_error = self.V_td_target - self.V_evaluated
            self.policy_loss_local = -tf.reduce_mean( tf.multiply( log_probs, self.V_td_error ) )
            self.policy_loss = self.policy_loss_local


        self.policy_opt = tf.train.AdamOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.policy_loss)

    
    def create_dqn_policy(self):
        if self.para_set.MODEL_NAME in ['QMix','VDN']:
            self.target_max_qvals = tf.placeholder(tf.float32, [None], 'target_max_qvals')
            if self.para_set.MODEL_NAME == 'VDN':    
                self.loss_vdn_qmix = tf.reduce_mean(tf.square(self.vdn_g_main - self.target_max_qvals))  
                # self.loss_vdn_qmix = tf.norm(self.vdn_g_main - self.target_max_qvals, 2)  
            if self.para_set.MODEL_NAME == 'QMix':    
                self.loss_vdn_qmix = tf.reduce_mean(tf.square(self.QMix_g_main - self.target_max_qvals))
                # self.loss_vdn_qmix = tf.norm(self.QMix_g_main - self.target_max_qvals, 2)  
            self.overall_loss = self.loss_vdn_qmix
        
        if self.para_set.MODEL_NAME == 'QCombo':
            # self.Q_g_evaluated = tf.placeholder(tf.float32, [None], 'Q_g_evaluated'), this will change the dimension
            self.self_rank = tf.placeholder(tf.float32, [None, self.n_agents], 'weights_for_agents')
            temp = tf.multiply(self.self_rank, tf.reshape(self.Q_mix_agents_main, [-1, self.n_agents]))
            self.Q_mix_sum = tf.reduce_sum(temp, axis=1) 
            # self.loss_comb_Q = tf.reduce_mean(tf.square(self.Q_i_sum - self.Q_g_evaluated))
            self.loss_comb_Q = tf.norm(self.Q_mix_sum - self.qcombo_g_main, 2)
            # if self.para_set.MODEL_NAME in ['VDN']:
            #     self.overall_loss = self.loss_comb_Q
            self.loss_i_Q = tf.reduce_mean(tf.reduce_sum(tf.reshape(
                tf.square(self.Q_target - self.Q_mix_agents_main), [-1, self.n_agents]), axis=1))
                                      #individual loss, global loss, and difference loss
            self.overall_loss =  1.2*self.loss_i_Q+self.loss_g_Q+ 0.4*self.loss_comb_Q
        # self.overall_loss = self.loss_g_Q
        self.Q_overall_opt = tf.train.AdamOptimizer(self.lr_Q)
        self.Q_overall_op = self.Q_overall_opt.minimize(self.overall_loss)

    def create_summary(self):

        summaries_op = [tf.summary.scalar('overall_loss', self.overall_loss)]
        summaries_op.append(tf.summary.scalar('loss_individual', self.loss_i_Q))
        summaries_op.append(tf.summary.scalar('loss_global', self.loss_g_Q))        
        summaries_op.append(tf.summary.scalar('loss_combined', self.loss_comb_Q))
        op_variables1 = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_i_main')]
        op_variables2 = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qcombo_g_main')]
        op_variables = op_variables1+op_variables2
        for v in op_variables:
            summaries_op.append(tf.summary.histogram(v.op.name, v))
        grads = self.Q_overall_opt.compute_gradients(self.overall_loss, op_variables)
        for grad, var in grads:
            if grad is not None:
                summaries_op.append( tf.summary.histogram(var.op.name+'/gradient', grad) )
        self.summary_op = tf.summary.merge(summaries_op)


    def action_choose(self, self_obs, maps, last_actions, global_state, n_steps, epsilon, count, sess):

        ''' choose the best action for current self_allsame_obs '''
        ###greedy choice, for Q learning, not good for coutinuous states###
        self_agent_labels = np.tile(self.agent_labels, (n_steps,1))

        _, self_obs_others, self_allsame_obs = self.process_global_obs(self_obs, n_steps)
        self_all_obs = np.concatenate((self_obs, self_obs_others),axis=1)

        if n_steps == 1:
            self_last_actions, self_last_actions_others = self.process_actions(n_steps, last_actions)
        else:
            self_last_actions = last_actions

        
        self_global_state = np.repeat(np.reshape(global_state,(n_steps, self.global_state_dimension)), self.n_agents, axis=0)
        temp = np.array([[1,0,0,0]])
        self_sub_labels = np.repeat(temp, self.n_agents*n_steps, axis=0)
        
        if self.para_set.MODEL_NAME in ['COMA','IAC']: 
            feed = {self.self_obs: self_obs, 
                    self.self_obs_others:self_obs_others, 
                    self.self_last_actions:self_last_actions,  
                    self.self_agent_labels: self_agent_labels,
                    self.epsilon:epsilon}
            action_samples_res = sess.run(self.action_samples_target_actor, feed_dict=feed)

            actions = np.reshape(action_samples_res, action_samples_res.shape[0])
        if self.para_set.MODEL_NAME in ['IDQN', 'QCombo', 'VDN', 'QMix']: 
            feed = {self.self_obs:self_obs, 
                self.self_obs_others:self_obs_others, 
                self.epsilon:epsilon, 
                self.self_agent_labels : self_agent_labels,
                self.self_global_state:self_global_state,
                self.self_last_actions: self_last_actions,
                self.self_sub_labels:self_sub_labels,
                self.n_steps:n_steps}
            action_samples_res = sess.run(self.action_samples, feed_dict=feed)
            actions_samples = np.reshape(action_samples_res, action_samples_res.shape[0])
            if random.random() < epsilon:  # continue explore new Random Action
                actions = np.random.choice([0,1], size=action_samples_res.shape[0], p=np.ones(2)*0.5)
                # print("##Explore")
            else:  # exploitation
                actions = actions_samples
        return actions

    ##this is the rnn choose 
    def rnn_choose(self, self_obs, maps, last_actions, global_state, the_pv, n_steps, epsilon, count, sess):

        ''' choose the best action for current self_allsame_obs '''
        ###greedy choice, for Q learning, not good for coutinuous states###
        
        self_agent_labels = np.tile(self.agent_labels, (n_steps,1))

        _, self_obs_others, self_allsame_obs = self.process_global_obs(self_obs, n_steps)
        self_all_obs = np.concatenate((self_obs, self_obs_others),axis=1)

        if n_steps == 1:
            self_last_actions, self_last_actions_others = self.process_actions(n_steps, last_actions)
        else:
            self_last_actions = last_actions
        
        self_global_state = np.repeat(np.reshape(global_state,(n_steps, self.global_state_dimension)), self.n_agents, axis=0)
        temp = np.array([[1,0,0,0]])
        self_sub_labels = np.repeat(temp, self.n_agents*n_steps, axis=0)

        if self.para_set.MODEL_NAME in ['COMA','IAC']: 
            feed = {self.self_obs: self_obs, 
                    self.self_obs_others:self_obs_others, 
                    self.self_last_actions:self_last_actions,  
                    self.self_agent_labels: self_agent_labels,
                    self.prev_state_target: the_pv.astype(float),
                    self.epsilon:epsilon,
                    self.n_steps:n_steps}
            pv, action_samples_res = sess.run([self.out_state_target, self.action_samples_target_actor], feed_dict=feed)

            actions = np.reshape(action_samples_res, action_samples_res.shape[0])
            
        if self.para_set.MODEL_NAME in ['IDQN','VDN', 'QCombo', 'QMix']:
            feed = {self.self_obs:self_obs, 
                    self.self_obs_others:self_obs_others, 
                    self.epsilon:epsilon, 
                    self.self_agent_labels: self_agent_labels,
                    self.self_global_state:self_global_state,
                    self.self_last_actions: self_last_actions,
                    self.self_sub_labels:self_sub_labels,
                    self.prev_state: the_pv.astype(float), 
                    self.n_steps:n_steps} 
            pv,action_samples_res = sess.run([self.out_state, self.action_samples], feed_dict=feed)

            actions_samples = np.reshape(action_samples_res, action_samples_res.shape[0])
            if random.random() < epsilon:  # continue explore new Random Action
                actions = np.random.choice([0,1], size=action_samples_res.shape[0], p=np.ones(2)*0.5)
                # print("##Explore")
            else:  # exploitation
                actions = actions_samples
        
        return actions, pv

    def run_actor_target(self, self_obs, maps, last_actions, n_steps,  epsilon, sess):
        """
        Gets actions from the slowly-updating policy
        """
    
        self_agent_labels = np.tile(self.agent_labels, (n_steps,1))

        _, self_obs_others, self_allsame_obs = self.process_global_obs(self_obs, n_steps)
        self_all_obs = np.concatenate((self_obs, self_obs_others),axis=1)

        # self_obs_others = np.reshape(self_obs_others, (n_steps*self.n_agents, self.self_obs_dimension))
        # _, self_obs_others, obs = self.process_global_obs(self_obs_others, n_steps)
        
        if n_steps == 1:
            self_last_actions, self_last_actions_others = self.process_actions(n_steps, last_actions)
        else:
            self_last_actions = last_actions
        

        feed = {self.self_obs: self_obs, 
                self.self_obs_others:self_obs_others, 
                self.self_last_actions:self_last_actions,  
                self.self_agent_labels: self_agent_labels,
                self.epsilon:epsilon}
        action_samples_res = sess.run(self.action_samples_target_actor, feed_dict=feed)
        
        
        return np.reshape(action_samples_res, action_samples_res.shape[0])
    
    def get_assign_target_ops(self, list_vars):

        # ops for equating main and target
        list_initial_ops = []
        # ops for slow update of target toward main
        list_update_ops = []

        # Repeat for Q if needed
        if self.para_set.MODEL_NAME in ['IDQN','QMix','QCombo','VDN']:
            
            list_Q_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_i_main')
            map_name_Q_main = {v.name.split('main')[1] : v for v in list_Q_main}
            list_Q_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_i_target')
            map_name_Q_target = {v.name.split('target')[1] : v for v in list_Q_target}

            if len(list_Q_main) != len(list_Q_target):
                raise ValueError("get_initialize_target_ops : lengths of Q_main and Q_target do not match")

            # ops for equating main and target
            for name, var in map_name_Q_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_Q_target[name].assign(var) )

            # ops for slow update of target toward main
            for name, var in map_name_Q_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_Q_target[name].assign( self.tau*var + (1-self.tau)*map_name_Q_target[name] ) )

        if self.para_set.MODEL_NAME in ['QCombo']:
            list_Mix_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qcombo_g_main')
            map_name_Mix_main = {v.name.split('main')[1] : v for v in list_Mix_main}
            list_Mix_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qcombo_g_target')
            map_name_Mix_target = {v.name.split('target')[1] : v for v in list_Mix_target}

            if len(list_Mix_main) != len(list_Mix_target):
                raise ValueError("get_initialize_target_ops : lengths of Q_main and Q_target do not match")

            # ops for equating main and target
            for name, var in map_name_Mix_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_Mix_target[name].assign(var) )

            # ops for slow update of target toward main
            for name, var in map_name_Mix_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_Mix_target[name].assign( self.tau*var + (1-self.tau)*map_name_Mix_target[name] ) )

        if self.para_set.MODEL_NAME in ['QMix']:
            list_QMix_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'QMix_main')
            map_name_QMix_main = {v.name.split('main')[1] : v for v in list_QMix_main}
            list_QMix_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'QMix_target')
            map_name_QMix_target = {v.name.split('target')[1] : v for v in list_QMix_target}

            if len(list_QMix_main) != len(list_QMix_target):
                raise ValueError("get_initialize_target_ops : lengths of Q_main and Q_target do not match")

            # ops for equating main and target
            for name, var in map_name_QMix_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_QMix_target[name].assign(var) )

            # ops for slow update of target toward main
            for name, var in map_name_QMix_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_QMix_target[name].assign( self.tau*var + (1-self.tau)*map_name_QMix_target[name] ) )

        # For policy
        if self.para_set.MODEL_NAME in ['IAC']:
            list_V_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_i_main')
            map_name_V_main = {v.name.split('main')[1] : v for v in list_V_main}
            list_V_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'V_i_target')
            map_name_V_target = {v.name.split('target')[1] : v for v in list_V_target}
            
            if len(list_V_main) != len(list_V_target):
                raise ValueError("get_initialize_target_ops : lengths of V_main and V_target do not match")
            
            for name, var in map_name_V_main.items():
                # create op that assigns value of main variable to
                # target variable of the same name
                list_initial_ops.append( map_name_V_target[name].assign(var) )
            
            for name, var in map_name_V_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_V_target[name].assign( self.tau*var + (1-self.tau)*map_name_V_target[name] ) )
        
        
        if self.para_set.MODEL_NAME in ['COMA']:
            list_coma_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_coma_main')
            map_name_coma_main = {v.name.split('main')[1] : v for v in list_coma_main}
            list_coma_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Q_coma_target')
            map_name_coma_target = {v.name.split('target')[1] : v for v in list_coma_target}

            if len(list_coma_main) != len(list_coma_target):
                raise ValueError("get_initialize_target_ops : lengths of coma_main and coma_target do not match")
            # ops for equating main and target
            for name, var in map_name_coma_main.items():
                list_initial_ops.append( map_name_coma_target[name].assign(var) )
            # ops for slow update of target toward main
            for name, var in map_name_coma_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_coma_target[name].assign( self.tau*var + (1-self.tau)*map_name_coma_target[name] ) )
        
        
        if self.para_set.MODEL_NAME in ['IAC','COMA']:
            list_P_main = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_main')
            map_name_P_main = {v.name.split('main')[1] : v for v in list_P_main}
            list_P_target = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Policy_target')
            map_name_P_target = {v.name.split('target')[1] : v for v in list_P_target}

            if len(list_P_main) != len(list_P_target):
                raise ValueError("get_initialize_target_ops : lengths of P_main and P_target do not match")
            # ops for equating main and target
            for name, var in map_name_P_main.items():
                list_initial_ops.append( map_name_P_target[name].assign(var) )
            # ops for slow update of target toward main
            for name, var in map_name_P_main.items():
                # incremental update of target towards main
                list_update_ops.append( map_name_P_target[name].assign( self.tau*var + (1-self.tau)*map_name_P_target[name] ) )
        return list_initial_ops, list_update_ops


    def process_actions(self, n_steps, actions):
        """
        actions must have shape [time, agents],
        and values are action indices
        """
        # Each row of actions is one time step,
        # row contains action indices for all agents
        # Convert to [time, agents, l_action]
        # so each agent gets its own 1-hot row vector
        self_actions = np.zeros([n_steps, self.n_agents, self.n_actions], dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        # actions: [time, n_agents], 
        self_actions[grid[0], grid[1], actions] = 1
        # Convert to format [time*agents, agents-1, l_action]
        # so that the set of <n_agent> actions at each time step
        # is duplicated <n_agent> times, and each duplicate
        # now contains all <n_agent>-1 actions representing
        # the OTHER agents actions
        # Not sure how to vectorize this operation, so here is
        # a brute force method
        list_to_interleave = []
        for n in range(self.n_agents):
            # extract all actions except agent n's action
            list_to_interleave.append( self_actions[:, np.arange(self.n_agents)!=n, :] )
        # interleave
        self_actions_others = np.zeros([self.n_agents*n_steps, self.n_agents-1, self.n_actions])
        for n in range(self.n_agents):
            self_actions_others[n::self.n_agents, :, :] = list_to_interleave[n]
        # In-place reshape of actions to [time*n_agents, l_action]
        self_actions.shape = (n_steps*self.n_agents, self.n_actions)

        return self_actions, self_actions_others


    def process_batch(self, batch):
        """
        Extract quantities of the same type from batch.
        Format batch so that each agent at each time step is one
        batch entry.
        Duplicate global quantities <n_agents> times to be
        compatible with this scheme.
        """
        # shapes are [time, ...original dims...]

        lanes_info = np.stack(batch[:,0]) #[time, agents, self.self_obs_dimension]
        maps = np.stack(batch[:,1]) #[time, agents, ?,?]
        global_state = np.stack(batch[:, 2]) #[time, 12]
        actions = np.stack(batch[:,3]) #[time, agents]
        local_rewards = np.stack(batch[:,4]) #[time, agents]
        global_reward = np.stack(batch[:, 5]) #[time] 
        next_lanes_info = np.stack(batch[:,6]) #[time, agents, 2, 3]
        next_maps = np.stack(batch[:,7]) #[time, agents, 3]
        next_global_state = np.stack(batch[:, 8]) #[time, 12]
        last_actions = np.stack(batch[:,9]) #[time, agents]
        done = np.stack(batch[:,10])#[time]
        

        # Try to free memory
        batch = None
    
        n_steps = lanes_info.shape[0]

        # For all global quantities, for each time step,
        # duplicate values <n_agents> times for
        # batch processing of all agents
        # v_global = np.repeat(v_global, self.n_agents, axis=0)
        # reward = np.repeat(reward, self.n_agents, axis=0)
        # v_global_next = np.repeat(v_global_next, self.n_agents, axis=0)
        done = np.repeat(done, self.n_agents, axis=0)

        # In-place reshape for *_local quantities,
        # so that one time step for one agent is considered
        # one batch entry
        lanes_info.shape = (n_steps*self.n_agents, self.self_obs_dimension)
        next_lanes_info.shape = (n_steps*self.n_agents, self.self_obs_dimension)
        
        #just ignore maps for now to quick processing
        # maps.shape = (n_steps*self.n_agents, np.size(maps,2), np.size(maps,3))
        # next_maps.shape = (n_steps*self.n_agents, np.size(next_maps,2), np.size(next_maps,3))
        
        local_rewards.shape = (n_steps*self.n_agents)

        self_actions, self_actions_others = self.process_actions(n_steps, actions)
        self_last_actions, self_last_actions_others = self.process_actions(n_steps, last_actions)
            
        return n_steps, lanes_info, maps, global_state, self_actions, self_actions_others, local_rewards, global_reward, next_lanes_info, next_maps, next_global_state, self_last_actions, done



    def process_global_obs(self, v_global, n_steps):
        """
        v_global has shape [n_steps, n_agents, l_state] [n_steps, n_agents*l_state]
        Convert to three streams:
        1. [n_agents * n_steps, state_one_agent] : each row is self_allsame_obs of one agent,
        and a block of <n_agents> rows belong to one sampled transition from batch
        2. [n_agents * n_steps, l_state_other_agents] : each row is the self_allsame_obs of all
        OTHER agents, as a single row vector. A block of <n_agents> rows belong to one 
        sampled transition from batch
        3. [n_steps*n_agents, n_agents*l_state] : each row is concatenation of self_allsame_obs of all agents
        For each time step, the row is duplicated <n_agents> times, since the same self_allsame_obs s is used
        in <n_agents> different evaluations of Q(s,a^{-n},a^n)
        """
        # Reshape into 2D, each block of <n_agents> rows correspond to one time step
        
        v_global_one_agent = np.reshape(v_global, (n_steps*self.n_agents, self.self_obs_dimension))
        v_global_others = np.zeros((n_steps*self.n_agents, self.n_agents-1, self.self_obs_dimension))
        temp_v_global = np.reshape(v_global, (n_steps, self.n_agents, self.self_obs_dimension))
        for n in range(self.n_agents):
            v_global_others[n::self.n_agents, :, :] = temp_v_global[:, np.arange(self.n_agents)!=n, :]
        # Reshape into 2D, each row is self_allsame_obs of all other agents, each block of
        # <n_agents> rows correspond to one time step
        v_global_others.shape = (n_steps*self.n_agents, (self.n_agents-1)*self.self_obs_dimension)
        v_global_concated = np.reshape(v_global, (n_steps, self.n_agents*self.self_obs_dimension))
        # v_global_concated = v_global
        self_allsame_obs = np.repeat(v_global_concated, self.n_agents, axis=0)
        return v_global_one_agent, v_global_others, self_allsame_obs

    


    def train_step(self, sess, batch, epsilon, idx_train, 
                   summarize=False, writer=None):

        # Each agent for each time step is now a batch entry
        n_steps, self_obs, maps, global_state, self_actions, self_actions_others, reward_local, reward_global, self_obs_next, maps_next, global_state_next, self_last_actions, done = self.process_batch(batch)
        
            
        _, self_obs_others, self_allsame_obs = self.process_global_obs(self_obs, n_steps)
        self_all_obs = np.concatenate((self_obs, self_obs_others),axis=1)
        # self_all_obs = np.concatenate((self_obs_others, self_obs_others, self_last_actions),axis=1)

        _, self_obs_others_next, self_allsame_obs_next = self.process_global_obs(self_obs_next, n_steps)
        self_all_obs_next = np.concatenate((self_obs_next, self_obs_others_next),axis=1)

        self_global_state = np.repeat(global_state, self.n_agents, axis=0)
        self_global_state_next = np.repeat(global_state_next, self.n_agents, axis=0)
 
        # last_actions_1hot_global = np.reshape(self_last_actions, (n_steps, self.n_agents*self.n_actions))
        # last_actions_1hot_next_global = np.reshape(self_actions, (n_steps, self.n_agents*self.n_actions))
        # global_state = np.concatenate((global_state, last_actions_1hot_global),axis=1)
        # global_state_next = np.concatenate((global_state_next, last_actions_1hot_global),axis=1)
        
        self_actions = self_actions.astype(float)
        # Create 1-hot agent labels [n_steps*n_agents, n_agents]
        self_agent_labels = np.tile(self.agent_labels, (n_steps,1))

        # ------------ Train local DQN ----------------#
        # actions: [time*n_agents], the greedy choice for the next target actions
        if self.para_set.MODEL_NAME in ['IDQN','QCombo','VDN','QMix']: 
            self_reward_global = np.repeat(reward_global, self.n_agents, axis=0)

            
            feed = {self.self_obs:self_obs_next, 
                    self.self_obs_others:self_obs_others_next, 
                    self.self_global_state : self_global_state_next,
                    self.epsilon:epsilon,
                    self.self_agent_labels : self_agent_labels,
                    self.self_last_actions: self_actions, 
                    self.n_steps: n_steps}

            #either choose the Q_i_target or choose the Rnn target or coma target
            Q_target_res = sess.run(self.Q_i_target, feed_dict=feed) #[batch*n_agents, 2] 2 is the number of actions
            action_samples_res = np.argmax(Q_target_res,axis=1)
            actions = action_samples_res
            # actions_r: [batch, n_agnets]
            actions_r = np.reshape(actions, [n_steps, self.n_agents])
            #self_actions_next[batch*agents, n_actions]
            self_actions_next, self_actions_others_next = self.process_actions(n_steps, actions_r)


            #--------Obtain the self Q through the RNN------------#  
            # stepsize = 2 
            # Q_target_res = []
            # for i in range(0, 30, stepsize):
            #     j = i + stepsize 
            #     k = i * self.n_agents 
            #     m = j * self.n_agents
            #     feed = {self.self_obs:self_obs_next[k:m], self.self_obs_others:self_obs_next[k:m], 
            #         self.epsilon:epsilon, self.n_steps:j-i}
            #     Q_target_res_i = sess.run(self.Q_i_target, feed_dict=feed) #[batch*n_agents, 2] 2 is the number of actions
            #     Q_target_res.append(Q_target_res_i) 
            # Q_target_res = np.stack(Q_target_res)
            # Q_target_res = np.reshape(Q_target_res, [n_steps*self.n_agents, self.n_actions])
            # actions_1 = self.dqn_choose(self_obs_next, self_obs_next, 0, idx_train, sess)
            # feed = {self.self_obs : self_obs_next,
            #         self.self_obs_others : self_obs_next}
            # Q_target_res = sess.run(self.Q_i_target, feed_dict=feed) #[batch*n_agents, 2] 2 is the number of actions
            # Q_target_res0 = tf.reduce_max(Q_target_res, axis=1).eval(session=sess) #[batch*n_agents]
            Q_target_res = np.sum(Q_target_res * self_actions_next, axis=1) #[batch*n_agents]
            done_multiplier = -(done - 1)
            # if true, then 0, else 1

        if self.para_set.MODEL_NAME in ('IDQN'):  
            
            Q_i_target = reward_local + self.gamma * Q_target_res * done_multiplier #[batch*n_agents] a very long list 
        
        # if self.para_set.MODEL_NAME == 'IDQN' or self.para_set.STAGE == 2 : 
            feed = {self.Q_target : Q_i_target,
                    self.self_actions : self_actions,
                    self.self_obs : self_obs,
                    self.self_obs_others : self_obs_others,
                    self.self_agent_labels : self_agent_labels,
                    self.self_last_actions: self_last_actions}
                    
            # Run optimizer for local DQN
            
            _ = sess.run(self.Q_op, feed_dict=feed)
        # ------------ Train QCombo Algo ----------------#
        # reshape the actions from [time*n_agents, n_actions]
        # Now each row is one time step, containing action
        # indices for all agents actions_r[time, n_agents]

        
        if self.para_set.MODEL_NAME in ['QCombo']: 
            
            #Q_target_res [batch*n_agents]
            Q_target_res = np.reshape(Q_target_res, [n_steps, self.n_agents])
            
            rank = np.repeat(self.rank, n_steps, axis=0)
            rank = np.reshape(rank, [n_steps, self.n_agents])
            Q_target_res = np.multiply(rank, Q_target_res)
            Q_target_res = np.reshape(Q_target_res, [n_steps*self.n_agents])
            # if true, then 0, else 1
            Q_i_target = reward_local + self.gamma * Q_target_res * done_multiplier #[batch*n_agents] a very long list 
            
            
            feed = {self.global_state : global_state_next, 
                    self.self_actions: self_actions_next}
            #Q_g_bellman[time]
            qcombo_g_target = sess.run(self.qcombo_g_target, feed_dict=feed)[0]
            Q_g_bellman = reward_global + self.gamma * qcombo_g_target * done_multiplier[0]
           

            feed = {self.Q_g_bellman: Q_g_bellman, 
                    self.global_state: global_state, 
                    self.self_global_state : self_global_state,
                    self.self_actions: self_actions,
                    self.Q_target: Q_i_target, 
                    self.self_obs : self_obs,
                    self.self_rank: rank, 
                    self.self_obs_others : self_obs_others,
                    self.self_last_actions: self_last_actions, 
                    self.self_agent_labels : self_agent_labels,
                    self.n_steps: n_steps}  
            if summarize:
                op_summary, _ = sess.run([self.summary_op, self.Q_overall_op], feed_dict=feed)
                writer.add_summary(op_summary, idx_train)
            else:
                _ = sess.run(self.Q_overall_op, feed_dict=feed)

        #-------Train the QMix VDN---------#
        
        ##starting the VDN
        if self.para_set.MODEL_NAME in ['VDN']:
            feed = {self.self_agent_qs:Q_target_res}
            target_max_qvals = sess.run(self.vdn_g_target, feed_dict=feed) #[batch]
            targets = reward_global + self.gamma * target_max_qvals * done_multiplier[0]
        
            feed = {self.target_max_qvals: targets, 
                    self.self_obs:self_obs, 
                    self.self_obs_others:self_obs_others,
                    self.self_agent_labels : self_agent_labels,
                    self.self_actions: self_actions, 
                    self.self_last_actions : self_last_actions, 
                    self.n_steps: n_steps}
            # feed = {self.self_agent_qs: Q_mix_i_evaluated}
            _ = sess.run(self.Q_overall_op, feed_dict=feed)
            # chosen_action_qvals = sess.run(self.vdn_g, feed_dict=feed) #[batch]

        ##starting the Qmix network
        if self.para_set.MODEL_NAME in ['QMix']:
            feed = {self.global_state: global_state_next,
                    self.self_agent_qs: Q_target_res}
            # this step just model the relationship between total q with individual q and global self_allsame_obs
            target_max_qvals = sess.run(self.QMix_g_target, feed_dict=feed) #[batch]
                #calculate 1-step Q-learning targets
            targets = reward_global + self.gamma * target_max_qvals * done_multiplier[0]    
            #--------Obtain the policy through the RNN------------#
            # for i in range(0, 30, stepsize):
            #     j = i + stepsize 
            #     k = i * self.n_agents 
            #     m = j * self.n_agents

            #     feed = {self.target_max_qvals: targets[i:j], 
            #             self.global_state: global_state[i:j],
            #             self.self_obs:self_obs[k:m], 
            #             self.self_obs_others:self_obs_others[k:m],
            #             self.self_actions: self_actions[k:m],
            #             self.n_steps:j-i}
        
            #     _ = sess.run(self.Q_overall_op, feed_dict=feed)
            #--------Obtain the policy through the FC------------#
            feed = {self.target_max_qvals: targets, 
                        self.global_state: global_state,
                        self.self_obs:self_obs, 
                        self.self_obs_others:self_obs_others,
                        self.self_agent_labels : self_agent_labels,
                        self.self_last_actions: self_last_actions, 
                        self.self_actions: self_actions,
                        self.n_steps: n_steps}
        
            _ = sess.run(self.Q_overall_op, feed_dict=feed)

            # print ("self_allsame_obs matrix", state_matrix)


        # ------------ Train local critic ----------------#
        if self.para_set.MODEL_NAME in ['IAC']:
            # V_target(o^n_{t+1}, g^n). V_next_res = V(o^n_{t+1},g^n) used in policy gradient
            feed = {self.self_obs : self_obs_next,
                    self.self_obs_others : self_obs_others_next,
                    self.self_last_actions:self_actions,  
                    self.self_agent_labels: self_agent_labels}
            V_target_res, V_next_res = sess.run([self.V_i_target, self.V_i_main], feed_dict=feed)
            V_target_res = np.squeeze(V_target_res)
            V_next_res = np.squeeze(V_next_res)
            # if true, then 0, else 1
            done_multiplier = -(done - 1)
            V_td_target = reward_local + self.gamma * V_target_res * done_multiplier
            
            # Run optimizer for local critic
            feed = {self.V_td_target : V_td_target,
                    self.self_obs_others : self_obs_others,
                    self.self_obs : self_obs, 
                    self.self_last_actions:self_last_actions,  
                self.self_agent_labels: self_agent_labels}
            _, V_res = sess.run([self.V_op, self.V_i_main], feed_dict=feed)

            # Already computed V_res when running V_op above
            V_res = np.squeeze(V_res)
            
            feed = {self.self_actions : self_actions.astype(float),
                    self.epsilon : epsilon,
                    self.self_obs_others : self_obs_others,
                    self.self_obs : self_obs,
                    self.self_last_actions:self_last_actions,  
                    self.self_agent_labels: self_agent_labels,
                    self.V_td_target : V_td_target,
                    self.V_evaluated : V_res}

            _ = sess.run(self.policy_op, feed_dict=feed)




        # ----------- Train coma ----------------------- #
        # -----------   The global critic -------------- #

        if self.para_set.MODEL_NAME in ['COMA']:
            
            # self_allsame_obs = np.repeat(global_state, self.n_agents, axis=0)
            # state_next = np.repeat(global_state_next, self.n_agents, axis=0)
            
            
            coma_reward = np.repeat(reward_global, self.n_agents, axis=0)
            
            actions = self.action_choose(self_obs_next, maps_next, self_actions, global_state_next, n_steps, epsilon, idx_train, sess)
            # Now each row is one time step, containing action
            # indices for all agents
            actions_r = np.reshape(actions, [n_steps, self.n_agents])
            self_actions_next, self_actions_others_next = self.process_actions(n_steps, actions_r)


            feed = {self.self_global_state : self_global_state_next,
                    self.self_actions_others : self_actions_others_next,
                    self.self_agent_labels : self_agent_labels,
                    self.self_obs_others : self_obs_others_next, 
                    self.self_obs: self_obs_next}
            Q_target_res = sess.run(self.Q_coma_target, feed_dict=feed)
            Q_target_res = np.sum(Q_target_res * self_actions_next, axis=1)

            # if true, then 0, else 1
            done_multiplier = -(done - 1)
            Q_td_target = coma_reward + self.gamma * Q_target_res * done_multiplier
            
            feed = {self.Q_td_target : Q_td_target,
                    self.self_actions : self_actions,
                    self.self_global_state : self_global_state,
                    self.self_actions_others : self_actions_others,
                    self.self_agent_labels : self_agent_labels,
                    self.self_obs_others : self_obs_others,
                    self.self_obs: self_obs}
            
                    
            # Run optimizer for global critic
            _ = sess.run(self.gc_Q_op, feed_dict=feed)

            #------------Train Policy---------------#            
            # feed = {self.global_state : self_allsame_obs,
            #         self.self_obs_others : self_all_obs,
            #         self.self_obs: self_obs, 
            #         self.self_actions_others : self_actions_others,
            #         self.self_agent_labels : self_agent_labels,
            #         self.epsilon : epsilon}

            feed = {self.self_global_state : self_global_state,
                    self.self_obs_others : self_obs_others,
                    self.self_obs: self_obs, 
                    self.self_actions_others : self_actions_others,
                    self.self_agent_labels : self_agent_labels,
                    self.epsilon : epsilon}
            Q_res, probs_res = sess.run([self.Q_coma, self.probs], feed_dict=feed)
        
            # feed = {self.self_actions : self_actions.astype(float),
            #         self.self_obs_others : self_all_obs,
            #         self.self_obs : self_obs,
            #         self.self_agent_labels : self_agent_labels,
            #         self.Q_evaluated : Q_res,
            #         self.epsilon : epsilon, 
            #         self.probs_evaluated : probs_res}

            feed = {self.self_actions : self_actions.astype(float),
                    self.self_obs_others : self_obs_others,
                    self.self_obs : self_obs,
                    self.self_agent_labels : self_agent_labels,
                    self.Q_evaluated : Q_res,
                    self.epsilon : epsilon, 
                    self.probs_evaluated : probs_res}

            _ = sess.run(self.policy_op, feed_dict=feed)

        sess.run(self.list_update_target_ops)



    ###the rnn part for evaluation
    def train_step_rnn(self, sess, batch, epsilon, idx_train, the_pv, 
                summarize=False, writer=None):

        # Each agent for each time step is now a batch entry
        n_steps, self_obs, maps, global_state, self_actions, self_actions_others, reward_local, reward_global, self_obs_next, maps_next, global_state_next, self_last_actions, done = self.process_batch(batch)
        

            
        _, self_obs_others, self_allsame_obs = self.process_global_obs(self_obs, n_steps)
        self_all_obs = np.concatenate((self_obs, self_obs_others),axis=1)
        # self_all_obs = np.concatenate((self_obs_others, self_obs_others, self_last_actions),axis=1)

        _, self_obs_others_next, self_allsame_obs_next = self.process_global_obs(self_obs_next, n_steps)
        self_all_obs_next = np.concatenate((self_obs_next, self_obs_others_next),axis=1)

        self_global_state = np.repeat(global_state, self.n_agents, axis=0)
        self_global_state_next = np.repeat(global_state_next, self.n_agents, axis=0)

        
        
        self_actions = self_actions.astype(float)
        # Create 1-hot agent labels [n_steps*n_agents, n_agents]
        self_agent_labels = np.tile(self.agent_labels, (n_steps,1))

        #-------Train the QMix VDN---------#
        if self.para_set.MODEL_NAME in ['IDQN','QCombo','VDN','QMix']: 
            self_reward_global = np.repeat(reward_global, self.n_agents, axis=0)

            feed = {self.self_obs:self_obs_next, 
                    self.self_obs_others:self_obs_others_next, 
                    self.self_global_state : self_global_state_next,
                    self.epsilon:epsilon,
                    self.self_agent_labels : self_agent_labels,
                    self.self_last_actions: self_actions, 
                    self.prev_state_target: the_pv.astype(float),
                    self.n_steps: n_steps}

            #either choose the Q_i_target or choose the Rnn target or coma target
            Q_target_res = sess.run(self.Q_i_target, feed_dict=feed) #[batch*n_agents, 2] 2 is the number of actions
            action_samples_res = np.argmax(Q_target_res,axis=1)
            actions = action_samples_res
            # actions_r: [batch, n_agnets]
            actions_r = np.reshape(actions, [n_steps, self.n_agents])
            #self_actions_next[batch*agents, n_actions]
            self_actions_next, self_actions_others_next = self.process_actions(n_steps, actions_r)


            #--------Obtain the self Q through the RNN------------#  
            # stepsize = 2 
            # Q_target_res = []
            # for i in range(0, 30, stepsize):
            #     j = i + stepsize 
            #     k = i * self.n_agents 
            #     m = j * self.n_agents
            #     feed = {self.self_obs:self_obs_next[k:m], self.self_obs_others:self_obs_next[k:m], 
            #         self.epsilon:epsilon, self.n_steps:j-i}
            #     Q_target_res_i = sess.run(self.Q_i_target, feed_dict=feed) #[batch*n_agents, 2] 2 is the number of actions
            #     Q_target_res.append(Q_target_res_i) 
            # Q_target_res = np.stack(Q_target_res)
            # Q_target_res = np.reshape(Q_target_res, [n_steps*self.n_agents, self.n_actions])
            # actions_1 = self.dqn_choose(self_obs_next, self_obs_next, 0, idx_train, sess)
            # feed = {self.self_obs : self_obs_next,
            #         self.self_obs_others : self_obs_next}
            # Q_target_res = sess.run(self.Q_i_target, feed_dict=feed) #[batch*n_agents, 2] 2 is the number of actions
            # Q_target_res0 = tf.reduce_max(Q_target_res, axis=1).eval(session=sess) #[batch*n_agents]
            Q_target_res = np.sum(Q_target_res * self_actions_next, axis=1) #[batch*n_agents]
            done_multiplier = -(done - 1)
            # if true, then 0, else 1

        if self.para_set.MODEL_NAME in ('IDQN'):  
            
            Q_i_target = reward_local + self.gamma * Q_target_res * done_multiplier #[batch*n_agents] a very long list 
        
        # if self.para_set.MODEL_NAME == 'IDQN' or self.para_set.STAGE == 2 : 
            feed = {self.Q_target : Q_i_target,
                    self.self_actions : self_actions,
                    self.self_obs : self_obs,
                    self.self_obs_others : self_obs_others,
                    self.self_agent_labels : self_agent_labels,
                    self.prev_state: the_pv.astype(float),
                    self.self_last_actions: self_last_actions,
                    self.n_steps:n_steps}
                    
            # Run optimizer for local DQN
            pv, _ = sess.run([self.out_state, self.Q_op], feed_dict=feed)
        # ------------ Train QCombo Algo ----------------#
        # reshape the actions from [time*n_agents, n_actions]
        # Now each row is one time step, containing action
        # indices for all agents actions_r[time, n_agents]

        
        if self.para_set.MODEL_NAME in ['QCombo']: 
            
            #Q_target_res [batch*n_agents]
            Q_target_res = np.reshape(Q_target_res, [n_steps, self.n_agents])
            
            rank = np.repeat(self.rank, n_steps, axis=0)
            rank = np.reshape(rank, [n_steps, self.n_agents])
            Q_target_res = np.multiply(rank, Q_target_res)
            Q_target_res = np.reshape(Q_target_res, [n_steps*self.n_agents])
            # if true, then 0, else 1
            Q_i_target = reward_local + self.gamma * Q_target_res * done_multiplier #[batch*n_agents] a very long list 
            
            
            feed = {self.global_state : global_state_next, 
                    self.self_actions: self_actions_next}
            #Q_g_bellman[time]
            qcombo_g_target = sess.run(self.qcombo_g_target, feed_dict=feed)[0]
            Q_g_bellman = reward_global + self.gamma * qcombo_g_target * done_multiplier[0]
           

            feed = {self.Q_g_bellman: Q_g_bellman, 
                    self.global_state: global_state, 
                    self.self_global_state : self_global_state,
                    self.self_actions: self_actions,
                    self.Q_target: Q_i_target, 
                    self.self_obs : self_obs,
                    self.self_rank: rank, 
                    self.self_obs_others : self_obs_others,
                    self.prev_state: the_pv.astype(float),
                    self.self_last_actions: self_last_actions, 
                    self.self_agent_labels : self_agent_labels,
                    self.n_steps: n_steps}  
            if summarize:
                op_summary, pv, _ = sess.run([self.summary_op, self.out_state, self.Q_overall_op], feed_dict=feed)
                writer.add_summary(op_summary, idx_train)
            else:
                pv, _ = sess.run([self.out_state, self.Q_overall_op], feed_dict=feed)

        ##starting the VDN
        if self.para_set.MODEL_NAME in ['VDN']:
            feed = {self.self_agent_qs:Q_target_res}
            target_max_qvals = sess.run(self.vdn_g_target, feed_dict=feed) #[batch]
            targets = reward_global + self.gamma * target_max_qvals * done_multiplier[0]
        
            feed = {self.target_max_qvals: targets, 
                    self.self_obs:self_obs, 
                    self.self_obs_others:self_obs_others,
                    self.self_agent_labels : self_agent_labels,
                    self.self_actions: self_actions, 
                    self.self_last_actions : self_last_actions, 
                    self.prev_state: the_pv.astype(float),
                    self.n_steps: n_steps}
            # feed = {self.self_agent_qs: Q_mix_i_evaluated}
            pv, _ = sess.run([self.out_state, self.Q_overall_op], feed_dict=feed)
            # chosen_action_qvals = sess.run(self.vdn_g, feed_dict=feed) #[batch]

        ##starting the Qmix network
        if self.para_set.MODEL_NAME in ['QMix']:
            feed = {self.global_state: global_state_next,
                    self.self_agent_qs: Q_target_res}
            # this step just model the relationship between total q with individual q and global self_allsame_obs
            target_max_qvals = sess.run(self.QMix_g_target, feed_dict=feed) #[batch]
            #calculate 1-step Q-learning targets
            targets = reward_global + self.gamma * target_max_qvals * done_multiplier[0]    
            #--------Obtain the policy through the RNN------------#
            # for i in range(0, 30, stepsize):
            #     j = i + stepsize 
            #     k = i * self.n_agents 
            #     m = j * self.n_agents

            #     feed = {self.target_max_qvals: targets[i:j], 
            #             self.global_state: global_state[i:j],
            #             self.self_obs:self_obs[k:m], 
            #             self.self_obs_others:self_obs_others[k:m],
            #             self.self_actions: self_actions[k:m],
            #             self.n_steps:j-i}
        
            #     _ = sess.run(self.Q_overall_op, feed_dict=feed)
            #--------Obtain the policy through the FC------------#
            feed = {self.target_max_qvals: targets, 
                        self.global_state: global_state,
                        self.self_obs:self_obs, 
                        self.self_obs_others:self_obs_others,
                        self.self_agent_labels : self_agent_labels,
                        self.self_last_actions: self_last_actions, 
                        self.self_actions: self_actions,
                        self.prev_state: the_pv.astype(float),
                        self.n_steps: n_steps}
        
            pv, _ = sess.run([self.out_state, self.Q_overall_op], feed_dict=feed)

            # print ("self_allsame_obs matrix", state_matrix)
        
        # ------------ Train local critic ----------------#
        if self.para_set.MODEL_NAME in ['IAC']:
            # V_target(o^n_{t+1}, g^n). V_next_res = V(o^n_{t+1},g^n) used in policy gradient
            feed = {self.self_obs : self_obs_next,
                    self.self_obs_others : self_obs_others_next,
                    self.self_last_actions:self_actions,  
                    self.self_agent_labels: self_agent_labels}
            V_target_res, V_next_res = sess.run([self.V_i_target, self.V_i_main], feed_dict=feed)
            V_target_res = np.squeeze(V_target_res)
            V_next_res = np.squeeze(V_next_res)
            # if true, then 0, else 1
            done_multiplier = -(done - 1)
            V_td_target = reward_local + self.gamma * V_target_res * done_multiplier
            
            # Run optimizer for local critic
            feed = {self.V_td_target : V_td_target,
                    self.self_obs_others : self_obs_others,
                    self.self_obs : self_obs, 
                    self.self_last_actions:self_last_actions,  
                self.self_agent_labels: self_agent_labels}
            _, V_res = sess.run([self.V_op, self.V_i_main], feed_dict=feed)

            # Already computed V_res when running V_op above
            V_res = np.squeeze(V_res)
            
            feed = {self.self_actions : self_actions.astype(float),
                    self.epsilon : epsilon,
                    self.self_obs_others : self_obs_others,
                    self.self_obs : self_obs,
                    self.self_last_actions:self_last_actions,  
                    self.self_agent_labels: self_agent_labels,
                    self.prev_state: the_pv.astype(float),
                    self.V_td_target : V_td_target,
                    self.V_evaluated : V_res,
                    self.n_steps:n_steps}

            pv, _ = sess.run([self.out_state, self.policy_op], feed_dict=feed)

        # ----------- Train coma ----------------------- #
        # -----------   The global critic -------------- #

        if self.para_set.MODEL_NAME in ['COMA']:
            
            # self_allsame_obs = np.repeat(global_state, self.n_agents, axis=0)
            # state_next = np.repeat(global_state_next, self.n_agents, axis=0)
            
            
            coma_reward = np.repeat(reward_global, self.n_agents, axis=0)
            
            actions,pv = self.rnn_choose(self_obs_next, maps_next, self_actions, global_state_next, the_pv, n_steps, epsilon, idx_train,sess)
            # Now each row is one time step, containing action
            # indices for all agents
            actions_r = np.reshape(actions, [n_steps, self.n_agents])
            self_actions_next, self_actions_others_next = self.process_actions(n_steps, actions_r)

            feed = {self.self_global_state : self_global_state_next,
                    self.self_actions_others : self_actions_others_next,
                    self.self_agent_labels : self_agent_labels,
                    self.self_obs_others : self_obs_others_next, 
                    self.self_obs: self_obs_next}
            Q_target_res = sess.run(self.Q_coma_target, feed_dict=feed)
            Q_target_res = np.sum(Q_target_res * self_actions_next, axis=1)

            # if true, then 0, else 1
            done_multiplier = -(done - 1)
            Q_td_target = coma_reward + self.gamma * Q_target_res * done_multiplier
            
            feed = {self.Q_td_target : Q_td_target,
                    self.self_actions : self_actions,
                    self.self_global_state : self_global_state,
                    self.self_actions_others : self_actions_others,
                    self.self_agent_labels : self_agent_labels,
                    self.self_obs_others : self_obs_others,
                    self.self_obs: self_obs}
            
                    
            # Run optimizer for global critic
            _ = sess.run(self.gc_Q_op, feed_dict=feed)

            #------------Train Policy---------------#            
            # feed = {self.global_state : self_allsame_obs,
            #         self.self_obs_others : self_all_obs,
            #         self.self_obs: self_obs, 
            #         self.self_actions_others : self_actions_others,
            #         self.self_agent_labels : self_agent_labels,
            #         self.epsilon : epsilon}

            feed = {self.self_global_state : self_global_state,
                    self.self_obs_others : self_obs_others,
                    self.self_obs: self_obs, 
                    self.self_actions_others : self_actions_others,
                    self.self_agent_labels : self_agent_labels,
                    self.prev_state: the_pv.astype(float),
                    self.n_steps:n_steps,
                    self.epsilon : epsilon}
            Q_res, probs_res = sess.run([self.Q_coma, self.probs], feed_dict=feed)
        
            # feed = {self.self_actions : self_actions.astype(float),
            #         self.self_obs_others : self_all_obs,
            #         self.self_obs : self_obs,
            #         self.self_agent_labels : self_agent_labels,
            #         self.Q_evaluated : Q_res,
            #         self.epsilon : epsilon, 
            #         self.probs_evaluated : probs_res}

            feed = {self.self_actions : self_actions.astype(float),
                    self.self_obs_others : self_obs_others,
                    self.self_obs : self_obs,
                    self.self_agent_labels : self_agent_labels,
                    self.Q_evaluated : Q_res,
                    self.prev_state: the_pv.astype(float),
                    self.epsilon : epsilon, 
                    self.probs_evaluated : probs_res,
                    self.n_steps:n_steps}

            pv, _ = sess.run([self.out_state, self.policy_op], feed_dict=feed)


        sess.run(self.list_update_target_ops)
        return pv 
    
        






        
