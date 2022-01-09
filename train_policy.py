# -*- coding: utf-8 -*-

'''

author: zhi zhang

SEED: random number for initializing the experiment
setting_memo: the folder name for this experiment
    The conf, data files will should be placed in conf/setting_memo, data/setting_memo respectively
    The records, model files will be generated in records/setting_memo, model/setting_memo respectively

'''


import copy
import json
import shutil
import numpy as np 
import tensorflow as tf 
import re
from matplotlib import pyplot as plt

import os
import sys 
sys.path.append('../')
import time as pytime
import math
from a03deeplight_agent import Alg
from a02sumo_agent import SumoAgent
import replay_buffer
import replay_buffer_dual
import xml.etree.ElementTree as ET
import shutil
from collections import defaultdict
from matplotlib.ticker import FuncFormatter, MaxNLocator


# Create entire computational graph
# Creation of new trainable variables for new curriculum
# stage is handled by networks.py, given the stage number
# alg = alg_new.Alg(experiment, dimensions, stage, n_agents, lr_V=lr_V, lr_Q=lr_Q, lr_actor=lr_actor, use_Q=use_Q, use_V=use_V, alpha=alpha, nn=config['nn'], IAC=config['IAC'])
# print("Initialized computational graph")

# list_variables = tf.trainable_variables()
# if stage == 1 or restore_same_stage or train_from_nothing:
#     saver = tf.train.Saver()
# elif stage == 2:
#     to_restore = [v for v in list_variables if ('stage-%d'%stage not in v.name.split('/') and 'Policy_target' not in v.name.split('/'))]
#     saver = tf.train.Saver(to_restore)
# else:
#     # restore only those variables that were not
#     # just created at this curriculum stage
#     to_restore = [v for v in list_variables if 'stage-%d'%stage not in v.name.split('/')]
#     saver = tf.train.Saver(to_restore)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.set_random_seed(seed)
# sess = tf.Session(config=config)

# writer = tf.summary.FileWriter('../saved/%s' % dir_name, sess.graph)

# sess.run(tf.global_variables_initializer())
# print("Initialized variables")

# if train_from_nothing == 0:
#     print("Restoring variables from %s" % dir_restore)
#     saver.restore(sess, '../saved/%s/%s' % (dir_restore, model_name))
#     for var in list_variables:
#         if var.name == 'Policy/actor_branch1/kernel:0':
#             print(sess.run(var))

# # initialize target networks to equal main networks
# sess.run(alg.list_initialize_target_ops)

# # save everything without exclusion
# saver = tf.train.Saver(max_to_keep=None)

# epsilon = epsilon_start
# # For computing average over 100 episodes
# reward_local_century = np.zeros(n_agents)
# reward_global_century = 0




import random

import numpy as np

# from tensorflow import set_random_seed
# set_random_seed((SEED))
import json
import os
import time
import sys 

# ================================= only change these two ========================================

setting_memo = "one_run"

# ================================= only change these two ========================================

#setting the paths 
# code_dir = "/Users/zz/Downloads/trafficLightRL-upload"
code_dir = os.path.split(os.path.realpath(__file__))[0]
# changes the current working directory to the given path.
# It returns None in all the cases.
os.chdir(code_dir)

sys.path.append(code_dir)


#set up four experiments
# elif "2phase" in traffic_file[0]:
#     dic_exp["RUN_COUNTS"] = 72000
# elif "synthetic" in traffic_file[0]:
#     dic_exp["RUN_COUNTS"] = 216000  
# 

import six
import inspect


nn_dic = {'fc':0, 'rnn':1}
feature_dic = {'others':0,'no_others':1}


# helper method from `peewee` project to add metaclass
_METACLASS_ = '_metaclass_helper_'
def with_metaclass(meta, base=object):
    return meta(_METACLASS_, (base,), {})


class OuterMeta(type):
    def __new__(mcs, name, parents, dct):
        cls = super(OuterMeta, mcs).__new__(mcs, name, parents, dct)
        for klass in dct.values():
            if inspect.isclass(klass):
                print("Setting outer of '%s' to '%s'" % (klass, cls))
                klass.outer = cls

        return cls     

class TrafficLightDQN(with_metaclass(OuterMeta)):

    DIC_AGENTS = {"IDQN": Alg, "QCombo": Alg, "VDN": Alg, "QMix": Alg, "COMA": Alg, "IAC":Alg, "Static": Alg, "Random": Alg}

    NO_PRETRAIN_AGENTS = []

    class ParaSet:
        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                #assign object self with the attribute
                setattr(self, key, value)

    class Plotter(object):
        outer = None
        def __init__(self, outerinstance):
            self.out = outerinstance 
             

        def plot_old(self):
            memo = "one_run "
            for i, j, k in os.walk('records/one_run/{}'.format(memo)):
                break   
            filenames = []
            for f in j:
                if '_unequal_' in f:
                    filenames.append('unqual')
                elif '_equal_' in f:
                    filenames.append('equal')
                elif 'switch' in f:
                    filenames.append('switch')
                elif 'synthetic' in f:
                    filenames.append('synthetic')
                else:
                    filenames.append('general')
        
        def plot_driver(self):
            self.plot_rewards()
            self.phaseplot()
            for feature in ['queue_length','wait_time','delay']:
                self.plot_feature(feature)

        def plot_feature(self, featurename):
            inputfile = os.path.join(self.out.path_set.PATH_TO_OUTPUT, 'log_rewards.txt')
            steps = []
            centers = []
            features = []
            with open(inputfile, 'r') as f:
                lines = f.readlines() 
                for i, line in enumerate(lines):   
                    line = line.strip('\n')            
                    thisline = list(map(lambda x: str(x), line.split('\t')))
                    if i == 0:
                        index = thisline.index(featurename)
                    else: 
                        s0 = thisline[0]
                        s1 = thisline[1]
                        s2 = thisline[index]
                        
                        # s3 = thisline[2]
                        step = int(re.findall('\d+', s0)[0])
                        # phase = int(re.findall('\d+', s3)[0])
                        # center = s1
                        feature = re.findall("\d+(?:\.\d+)?", s2)
                        feature = float(feature[0]) 
                     
                        steps.append(step)
                        # centers.append(center)
                        features.append(feature)
                    # phases.append(phase)
            # iter*center*action, now convert to iter, center*action
            steps = np.array(steps)
            features = np.array(features)
            r_steps = np.reshape(steps, (-1, self.out.n_agents*5))
            r_features = np.reshape(features, (-1, self.out.n_agents*5))

            R = np.random.random_sample((1,))
            G = np.random.random_sample((1,))
            B = np.random.random_sample((1,))
            colors = list(zip(R, G, B))

            plot_steps = np.max(r_steps,axis=1)
            plot_features = np.average(r_features, axis=1)
            
            # start = self.out.para_set.PRETRAIN_EPSIODES
            # plot_features = plot_features[start:]
            # end = len(plot_steps)
            # diff = end - start 
            # plot_steps = range(diff)

            stride = 10 
            if len(plot_features) <= 10000:
                stride = 1        
            plot_steps = plot_steps[0::stride]
            plot_features = plot_features[0::stride] 
            plot_pointers = range(len(plot_steps))
            
            
            fig, ax = plt.subplots()
            def format_fn(tick_val, tick_pos):
                if int(tick_val) in plot_pointers:
                    return plot_steps[int(tick_val)]
                else:
                    return ''
            ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.plot(plot_pointers, plot_features, marker='', color=colors[0])
            ax.legend()
            plt.xlabel('#Time Stemps')
            label = featurename    
            # plt.ylabel(label)
            # plt.title("global "+featurename)
            plotfile = self.out.path_set.PATH_TO_OUTPUT+"/{}.png".format(featurename)
            fig.savefig(plotfile)

        def plot_rewards(self):
            inputfile = os.path.join(self.out.path_set.PATH_TO_OUTPUT, 'memories.txt')
            steps = []
            local_rewards = []
            global_rewards = []
            with open(inputfile, 'r') as f:
                lines = f.readlines() 
                for i, line in enumerate(lines):   
                    thisline = list(map(lambda x: str(x), line.split('\t')))
                    s0 = thisline[0]
                    s1 = thisline[1]
                    s2 = thisline[2]
                    # s3 = thisline[2]
                    step = int(re.findall('\d+', s0)[0])
                    # phase = int(re.findall('\d+', s3)[0])
                    local_reward = re.findall("-?\d+\.\d+", s1)
                    local_reward = [float(i) for i in local_reward]
                    if self.out.para_set.MODEL_NAME == "IDQN":
                        global_reward = sum(local_reward)/self.out.n_agents 
                    global_reward = re.findall("-?\d+\.\d+", s2)
                    global_reward = float(global_reward[0]) 
                     
                    steps.append(step)
                    local_rewards.append(local_reward)
                    global_rewards.append(global_reward)
                    # phases.append(phase)
            ### start drawing after training starts 
            # start = self.out.para_set.PRETRAIN_EPSIODES
            # local_rewards = local_rewards[start:]
            # global_rewards = global_rewards[start:]
            # end = len(steps)
            # diff = end - start 
            # steps = range(diff)
            
            ## stride if necessary
            stride = 10
            if len(global_rewards) <=1000:
                stride = 1        
            steps = steps[0::stride]
            local_rewards = local_rewards[0::stride]
            global_rewards = global_rewards[0::stride] 
            x_pointers = range(len(steps))
            def format_fn(tick_val, tick_pos):
                if int(tick_val) in x_pointers:
                    return steps[int(tick_val)]
                else:
                    return ''
            
            if self.out.para_set.PLOT['global_reward']: 
                fig, ax = plt.subplots()
                ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.plot(x_pointers, global_rewards, marker='')
                ax.legend()
                plt.xlabel("# Time Stamps")
                label = "rewards"    
                plt.ylabel(label)
                # plt.title('{}: global reward'.format(self.out.para_set.MODEL_NAME))
                plotfile = self.out.path_set.PATH_TO_OUTPUT+"/{}_global_rewards.png".format(self.out.para_set.MODEL_NAME)
                fig.savefig(plotfile)

            
            fig, ax = plt.subplots()
            R = np.random.random_sample((self.out.n_agents,))
            G = np.random.random_sample((self.out.n_agents,))
            B = np.random.random_sample((self.out.n_agents,))
            colors = list(zip(R, G, B))
            series = list(zip(*local_rewards))
            # series.pop(2)
            # ax.errorbar(nlist, dcs, yerr=e, marker='o')
            for i, (s, l) in enumerate(zip(series, self.out.agent_ids)):
                ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.plot(x_pointers, s, marker='', label=l, color=colors[i])
                ax.legend()
            label = "rewards"    
            plt.ylabel(label)
            # plotfolder = "records/one_run/{}/{}".format(memo, loc)
            # if not os.path.exists(plotfolder):
            #     os.makedirs(plotfolder)
            # plt.title('divide and conquer VS dynamic programming')
            plotfile = self.out.path_set.PATH_TO_OUTPUT+"/{}_local_rewards.png".format(self.out.para_set.MODEL_NAME)
            fig.savefig(plotfile)

        def phaseplot(self):
            stride = 100        
            inputfile = os.path.join(self.out.path_set.PATH_TO_OUTPUT, 'memories.txt')
            steps = []
            tfl_phases = []
            with open(inputfile, 'r') as f:
                lines = f.readlines() 
                for i, line in enumerate(lines):   
                    thisline = list(map(lambda x: str(x), line.split('\t')))
                    s0 = thisline[0]
                    s3 = thisline[3]
                    # s3 = thisline[2]
                    step = int(re.findall('\d+', s0)[0])
                    # phase = int(re.findall('\d+', s3)[0])
                    phase = re.findall("\d+", s3)
                    steps.append(step)
                    phase = [int(i) for i in phase]
                    tfl_phases.append(phase) 
                    
            stride = 1000 
            if len(tfl_phases) < 10000:
                stride = 1        
                    
            fig, ax = plt.subplots()
            R = np.random.random_sample((self.out.n_agents,))
            G = np.random.random_sample((self.out.n_agents,))
            B = np.random.random_sample((self.out.n_agents,))
            colors = list(zip(R, G, B))
            series = list(zip(*tfl_phases))
            # series.pop(2)
            # ax.errorbar(nlist, dcs, yerr=e, marker='o')
            for i, (s, l) in enumerate(zip(series, self.out.agent_ids)):
                
                denominator = list(np.ones((len(s))))
                nn = [sum(s[i:i+stride]) for i in range(0, len(s), stride)]
                nd = [sum(denominator[i:i+stride]) for i in range(0, len(denominator), stride)]
                ntimes = steps[0::stride]
                nseries = [x/y for x, y in zip(nn, nd)]
    
                
                ax.plot(ntimes, nseries, marker='', label=l, color=colors[i])
                ax.legend()
            label = "phases"    
            plt.ylabel(label)
            plotfile = self.out.path_set.PATH_TO_OUTPUT+"/{}_phases.png".format(self.out.para_set.MODEL_NAME)
            fig.savefig(plotfile)

            
            # inputfile = 'records/one_run/{}/{}/log_rewards.txt'.format(memo, loc)
            # times = []
            # qlength = []
            # delay = []
            # duration = []
            
            # with open(inputfile, 'r') as f:
            #     lines = f.readlines() 
            #     for i, line in enumerate(lines): 
            #         if i==0:
            #             names = list(map(lambda x: str(x), line.split(',')))
            #         else:   
            #             thisline = list(map(lambda x: float(x), line.split(',')))
            #             epic = thisline[0]
            #             s1 = thisline[11]
            #             s2 = thisline[2]
            #             s3 = thisline[3]
                        
            #             times.append(epic)
            #             qlength.append(s1)
            #             delay.append(s2)
            #             duration.append(s3)

            # phaseplot('phase', phases, times0, filename)
            
            # plot('queue length', qlength, times, filename)    
            # plot('duration', duration, times, filename)
            # plot('delay', delay, times, filename)

    class PathSet:

        # ======================================= conf files ========================================
        EXP_CONF = "exp.conf"
        SUMO_AGENT_CONF = "sumo_agent.conf"
        FLOW_CONF = "flow.conf"
        PATH_TO_CFG_TMP = os.path.join("data", "tmp")
        # ======================================= conf files ========================================

        def __init__(self, path_to_conf, path_to_data, path_to_output, path_to_model):

            self.PATH_TO_CONF = path_to_conf
            self.PATH_TO_DATA = path_to_data
            self.PATH_TO_OUTPUT = path_to_output
            self.PATH_TO_MODEL = path_to_model

            if not os.path.exists(self.PATH_TO_OUTPUT):
                os.makedirs(self.PATH_TO_OUTPUT)
            if not os.path.exists(self.PATH_TO_MODEL):
                os.makedirs(self.PATH_TO_MODEL)

            dic_paras = json.load(open(os.path.join(self.PATH_TO_CONF, self.EXP_CONF), "r"))
            self.AGENT_CONF = "{0}_agent.conf".format(dic_paras["MODEL_NAME"])
            pass

            # self.TRAFFIC_FILE = dic_paras["TRAFFIC_FILE"]
            # self.TRAFFIC_FILE_PRETRAIN = dic_paras["TRAFFIC_FILE_PRETRAIN"]


    def __init__(self, memo, algo_choice, f_prefix):

        self.path_set = self.PathSet(os.path.join("conf", memo),
                                     os.path.join("data", memo),
                                     os.path.join("records", memo, algo_choice),
                                     os.path.join("records", memo, algo_choice))
        conf_file = os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF)
        dic_paras = json.load(open(conf_file, "r"))
        #return key value pair to initiate the para_set
        self.para_set = self.ParaSet(dic_paras)
        train_test = self.para_set.TRAIN_TEST
        ##the flowname 
        fn = self.para_set.FLOWNAME
        self.rnn = self.para_set.RNN 

        if self.para_set.SUBPROB: 
            f_prefix = f_prefix + "_Sub"
            m_prefix = 'Train_(2,2)_Test_(2,2)'
            model_path = os.path.join(self.path_set.PATH_TO_OUTPUT, 'Train_(2,2)_Test_(2,2)', fn, "model")
        else:
            f_prefix = f_prefix 
            m_prefix = f_prefix 
            model_path = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, fn, "model") 

        train_path = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, fn, "train")
        
        if train_test not in ["train","test"]: #clean folder purpose 
            if os.path.exists(train_path): 
                # for i, j, k in os.walk(test_path):      
                    # break   
                # test_subdirs = [os.path.join(i, f) for f in j]
                # test_subdirs = j 
                #-----remove directories that are not existed in the test------
                for option in ['train','test']:
                    the_path = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, fn, option)
                    if os.path.exists(the_path):
                        for i, j, k in os.walk(the_path):      
                            break   
                        # train_subdirs = [os.path.join(i, f) for f in j]
                        the_subdirs = j 
                        the_subdirs = [f for f in j if f!='history']
                        # diff_subdirs = [j for j in train_subdirs if j not in test_subdirs] 
                        # for sub in train_subdirs
                        # if len(diff_subdirs) > 0:
                        for sub in the_subdirs:
                            # delete the unfinished folders
                            old_model_dir = os.path.join(self.path_set.PATH_TO_MODEL, f_prefix, fn, "model", sub)
                            old_record_dir = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, fn, option, sub)
                            if os.path.isfile(os.path.join(old_record_dir, "delay.png")) == False:
                                if os.path.exists(old_record_dir):
                                    shutil.rmtree(old_record_dir)
                                
                                if os.path.exists(old_model_dir):
                                    shutil.rmtree(old_model_dir)
                            #put the old folders into the history directory

                for path in [train_path, model_path]:           
                    for i, j, k in os.walk(path):      
                        break   
                    all_subdirs = [os.path.join(i, f) for f in j if f!='history']
                    all_subdirs.sort(key=lambda x: os.path.getmtime(x))
                    if truncated == 0:
                        zip_subdirs = all_subdirs 
                    else:
                        zip_subdirs = all_subdirs[:-truncated]    
                    dest = os.path.join(path, 'history')
                    self.create_dir(dest)
                    for zip_sub in zip_subdirs:
                        pick = os.path.split(zip_sub)[1]
                        dest_dir = os.path.join(dest, pick)
                        if os.path.exists(dest_dir):
                            shutil.rmtree(zip_sub)
                        else:
                            shutil.move(zip_sub, dest)         

            # if overwrite: 
            #     for i, j, k in os.walk(self.path_set.PATH_TO_MODEL):      
            #         break   
            #     all_subdirs = [os.path.join(i, f) for f in j]
            #     # Choose the one with the latest folder name 
            #     if len(all_subdirs) > 0:
            #         pick = os.path.split(max(all_subdirs, key=os.path.getmtime))[-1]
            #         # delte the record and model directory for this
            #         old_model_dir = os.path.join(self.path_set.PATH_TO_MODEL, pick)
            #         old_record_dir = os.path.join(self.path_set.PATH_TO_OUTPUT, pick)
            #         shutil.rmtree(old_model_dir)
            #         if os.path.exists(old_record_dir):
            #             shutil.rmtree(old_record_dir)
        else: #here the self.para_set got initialized 
            pass 
        # seed = name_options[0]
        # nn = name_options[1]
        # feature = feature_dic[name_options[2]]

        if train_test =="train":   

            #now construct the name
                     
            label_str = "T({0})_S({1})".format(time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())), SEED)

            # if nn:
            #     label_str = label_str+"-N()"
            #     -N({2})-F({3})    
            self.path_set.PATH_TO_OUTPUT = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, fn, "train", label_str)
            self.path_set.PATH_TO_MODEL = os.path.join(self.path_set.PATH_TO_MODEL, f_prefix, fn, "model", label_str)          

            self.create_dir(self.path_set.PATH_TO_OUTPUT)
            self.create_dir(self.path_set.PATH_TO_MODEL)
        if train_test =="test": 
            for i, j, k in os.walk(os.path.join(self.path_set.PATH_TO_MODEL, m_prefix, fn, "model")):      
                break   
            all_subdirs = [os.path.join(i, f) for f in j if f!='history']
            # Choose the one with the latest folder name 
            if len(all_subdirs) == 0:
                raise ValueError("please train before you test")  
            
            all_subdirs.sort(key=lambda x: os.path.getmtime(x))
            
            ### here implement the logic for checking the folder name for different options
            thelabel = str(runtime_seed[the_seed_key]) 
            picks = [os.path.split(i)[-1] for i in all_subdirs]
            for p in picks:
                # the_p_list = p.split("-")
                if thelabel in p:
                    pick = p
                    break  
            # pick = os.path.split(all_subdirs[-model_for_test])[-1]  
            # pick = os.path.split(max(all_subdirs, key=os.path.getmtime))[-1] 
            if self.para_set.MODEL_NAME in ['Random','Static']:
                label_str = "T({0})_S({1})".format(time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())), SEED)
                self.path_set.PATH_TO_OUTPUT = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, fn, "test", label_str)
                if not os.path.exists(self.path_set.PATH_TO_OUTPUT):
                    self.create_dir(self.path_set.PATH_TO_OUTPUT)
            else:
                self.path_set.PATH_TO_MODEL = os.path.join(self.path_set.PATH_TO_MODEL, m_prefix, fn, "model", pick)
                self.path_set.PATH_TO_OUTPUT = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, fn, "test", pick)
                if os.path.exists(self.path_set.PATH_TO_OUTPUT):
                    # num = int(self.path_set.PATH_TO_OUTPUT[-2]) + 1
                    thetime = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())) 
                    self.path_set.PATH_TO_OUTPUT = self.path_set.PATH_TO_OUTPUT + thetime 
                    os.makedirs(self.path_set.PATH_TO_OUTPUT)
                else:
                    self.create_dir(self.path_set.PATH_TO_OUTPUT)
        # else:
        #     self.path_set.PATH_TO_OUTPUT = os.path.join(self.path_set.PATH_TO_OUTPUT, f_prefix, "test")
        #     self.create_dir(self.path_set.PATH_TO_OUTPUT)
        
        if train_test in ['train','test']:
            
            if train_test=="train":
                shutil.copy(
                    os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF),
                    os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.EXP_CONF))
            
            for filename in ["a01traffic_light_dqn.py","a02sumo_agent.py",
                    "a03deeplight_agent.py","a04network_agent.py","analysis.py"]:
                shutil.copy(os.path.join(code_dir, filename),
                    os.path.join(self.path_set.PATH_TO_OUTPUT, filename))
        
            # reset the tensorflow graph every time for training without reuse the same variables again 
            tf.reset_default_graph()
            # starts the sumo environments 
            self.env = SumoAgent(self.path_set)
            self.agent_ids = self.env.get_tfl_ids()
            dimensions = {'self_obs':self.env.self_obs_dim, 'global_state': self.env.global_state_dim}
            file_name = os.path.join(self.path_set.PATH_TO_OUTPUT, "agent_ids.txt")
            f_memory = open(file_name, "a")
            f_memory.write("{}".format(self.agent_ids) + "\n")
            f_memory.close()
            self.n_agents = len(self.agent_ids)
            ## This will initialize the tensorflow network structure, use the path_set to point the algo algorithm option
            self.alg = self.DIC_AGENTS[self.para_set.MODEL_NAME](self.para_set, self.path_set, dimensions, 
                                        self.env.rank, n_agents=self.n_agents, num_actions=2 )
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.seed = SEED
            tf.set_random_seed(self.seed)
            self.sess = tf.Session(config=self.config)
            self.writer = tf.summary.FileWriter(self.path_set.PATH_TO_MODEL, self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
            print("Initialized variables")


    def check_if_need_pretrain(self):

        if self.para_set.MODEL_NAME in self.NO_PRETRAIN_AGENTS:
            return False
        else:
            return True

    def _generate_pre_train_ratios(self, phase_min_time, em_phase):
        """10, 0"""
        phase_traffic_ratios = [phase_min_time]

        # generate how many varients for each phase
        for i, phase_time in enumerate(phase_min_time):
            if i == em_phase:
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            else:
                # pass
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            for j in range(5, 20, 5):
                gen_phase_time = copy.deepcopy(phase_min_time)
                gen_phase_time[i] += j
                phase_traffic_ratios.append(gen_phase_time)

        return phase_traffic_ratios

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    def create_dir(self, dire):
        if not os.path.exists(dire):
            os.makedirs(dire)


    def train(self):

        # initialize target networks to equal main networks
    
        self.sess.run(self.alg.list_initialize_target_ops)
        # variables_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print ("Variable: ", k)
        #     print ("Shape: ", v.shape)
        #     print (v)

        # save everything without exclusion
        model_saver = tf.train.Saver(max_to_keep=None)
        epsilon_start = 0.9
        epsilon = epsilon_start
        # For computing average over 100 episodes
        reward_local_century = np.zeros(self.n_agents)
        reward_global_century = 0
        
        #Initialize the buffer
        dual_buffer = False
        if dual_buffer:
            # buf = replay_buffer_dual.Replay_Buffer(size=buffer_size, threshold=threshold)
            buf = replay_buffer_dual.Replay_Buffer(size=1000)
            num_good = 0
            num_bad = 0
            buf_episode = []
        else:
            buf = replay_buffer.Replay_Buffer(size=1000)

        total_run_cnt = self.para_set.RUN_COUNTS

        # initialize output streams
        file_name_memory = os.path.join(self.path_set.PATH_TO_OUTPUT, "memories.txt")
      
        # Starts sumo environment
        
        current_time = self.env.get_current_time()  # in seconds
        local_rewards = np.zeros(self.n_agents)
        last_actions = np.random.choice([0,1], size=self.n_agents, p=np.ones(2)*0.5)
            
        # sys.exit(0)
        epochs_period = self.para_set.EPOCHS_PERIOD 
        epsilon_step = epochs_period*(1 - 0.05)/float(5000)
        te_period = self.para_set.TRAIN_EVA_PERIOD
        # start experiment, there is no explict "done" in TFL
        pretrain_epsiodes = self.para_set.PRETRAIN_EPSIODES
        runcount=0
        fixed_idx=pretrain_epsiodes
        pre_fixed_idx=fixed_idx  
        pv = np.zeros((self.n_agents, 64), dtype='f')
        updating_count = 0 
        while current_time < total_run_cnt:

            
            f_memory = open(file_name_memory, "a")
            # get state, which will be used to get the actions
            lanes_info, maps, global_state, done = self.env.get_observation()
            maps = []
            
            ### the control index to control evaluating or training
            if current_time == fixed_idx: ## now pure training 
                pre_fixed_idx = fixed_idx #update the updating idx  
                fixed_idx = fixed_idx + te_period
                updating_count+=1 
                # print ("     algo", self.para_set.MODEL_NAME)  # the first 200 steps that training
            if current_time<pretrain_epsiodes or (0<=current_time - pre_fixed_idx<=0.5*te_period):
                eva = False 
                if updating_count == 1:
                    eva = True
                if (current_time< pretrain_epsiodes):
                    actions = np.random.choice([0,1], size=self.n_agents, p=np.ones(2)*0.5)
                else:
                    if self.rnn:
                        actions, pv = self.alg.rnn_choose(lanes_info, maps, last_actions, global_state, pv, 1, epsilon, current_time, self.sess) #[n_agents]  #[n_agents]
                    else:
                        actions = self.alg.action_choose(lanes_info, maps, last_actions, global_state, 1, epsilon, current_time, self.sess) #[n_agents]
        
                        
                # get reward from sumo agent, multiagent should be a vector data type 
                # multi-matrix, multi-vector, multi-matrix, single-scaler
                next_lanes_info, next_maps, next_global_state, local_rewards, global_reward, done = self.env.take_action(actions, current_time, pretrain_epsiodes, eva)
                # print ('     training rewards: local', local_rewards)
                
                # if current_time%100==0:
                # print ('     training rewards: global', global_reward)
                # cur_phases = global_state[:self.n_agents]
                # next_phases = global_state[self.n_agents:]
                # if current_time >= pretrain_epsiodes: record the training reward
                #     memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(current_time, local_rewards, global_reward)
                #     f_memory.write(memory_str + "\n")
                #     f_memory.close()
                
                current_time = self.env.get_current_time()
                
                    
                # remember
                # store transition into memory

                if dual_buffer:
                    buf_episode.append( np.array([np.array(lanes_info), np.array(maps), np.array(global_state), actions, np.array(local_rewards), np.array(global_reward),np.array(next_lanes_info), np.array(next_maps), np.array(next_global_state), last_actions, done]) )
                else:
                    buf.add( np.array([ np.array(lanes_info), np.array(maps), np.array(global_state), actions, np.array(local_rewards), np.array(global_reward),np.array(next_lanes_info), np.array(next_maps), np.array(next_global_state), last_actions, done])  )
                
                # threshold = -1000 
                # buf.add(buf_episode, np.sum(local_rewards) < threshold) 
                # The environment is infinite
                epochs = self.para_set.EPOCHS
                batch_size = self.para_set.BATCH_SIZE 
                period = 20 # every 20 periods write the first 
                epochs_period = self.para_set.EPOCHS_PERIOD 
                # epsilon_decay = (epsilon_start - 0.001)/(self.para_set.RUN_COUNTS-pretrain_epsiodes+1000)*epochs_period
                start_time = pytime.time()
                if (current_time>= pretrain_epsiodes):
                    # memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(
                    #     current_time-pretrain_epsiodes, local_rewards, global_reward)
                    # f_memory.write(memory_str + "\n")
                    # f_memory.close()
                    if updating_count == 1:
                        memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(current_time, local_rewards, global_reward)
                        f_memory.write(memory_str + "\n")
                        f_memory.close()
                    if (current_time % epochs_period == 0):#every epochs_period time step to train the neural network
                        if epsilon > 0.001:
                            epsilon = epsilon * 0.995
                            # epsilon -= epsilon_step
                            # if epsilon > 0.5:
                            #     # epsilon = epsilon * 0.997
                            #     epsilon = epsilon * 0.997
                            # else:
                            #     # epsilon = epsilon * 0.993
                            #     epsilon = epsilon * 0.993
                            # epsilon = epsilon - epsilon_decay
                        
                        # print ("     epsilon", epsilon) 
                        # print ('     rewards: global', global_reward)
                        ##if we want the rnn or the basic 
                        if self.rnn:
                            train_batch = 6*batch_size    #since 975/5 = 195, so can only holds 6 trainalbe state
                            for idx_epoch in range(6):
                                # Sample batch of transitions from replay buffer
                                #----sample_batch----, rnn needs sample consecutive samples# 
                                # if self.para_set.MODEL_NAME in ["QMix","VDN"]:
                                batch = buf.sample_rnn_batch(batch_size, train_batch, idx_epoch)
                                # if self.para_set.MODEL_NAME in ["QMix","VDN"]:
                                pv = self.alg.train_step_rnn(self.sess, batch, epsilon, current_time, pv, summarize=False, writer=None)
                        else: #do not use the rnn
                            for idx_epoch in range(epochs):
                                # Sample batch of transitions from replay buffer
                                #----sample_batch----, rnn needs sample consecutive samples# 
                                if self.para_set.MODEL_NAME in [""]:
                                    batch = buf.sample_batch_sequential(batch_size)
                                #----if not rnn used, then just random select from buffers-----
                                if self.para_set.MODEL_NAME in ["QMix","VDN","IDQN","IAC","QCombo","COMA"]:
                                    batch = buf.sample_batch(batch_size)
                                # print ('batch size', batch.shape[0])
                                # print ("train")
                                # Write TF summary every <period> episodes, for the first minibatch
                                if self.para_set.MODEL_NAME == 'QCombo':
                                    if idx_epoch == 0: 
                                        self.alg.train_step(self.sess, batch, epsilon, current_time, summarize=True, writer=self.writer)
                                    else:                                
                                        self.alg.train_step(self.sess, batch, epsilon, current_time, summarize=False, writer=None)
                                else:
                                    self.alg.train_step(self.sess, batch, epsilon, current_time, summarize=False, writer=None)
                # self.alg.train_step(self.sess, batch, epsilon, current_time, summarize=True, writer=None)
                        runtime = pytime.time()-start_time
                        # print ('     running time', runtime)
            else: 

                if self.rnn:
                    actions, pv, eva = self.evaluation_rnn(lanes_info, maps, last_actions, global_state, local_rewards, global_reward, current_time, pretrain_epsiodes, pre_fixed_idx, te_period, pv, f_memory, eva)
                else:
                    actions, eva = self.evaluation(lanes_info, maps, last_actions, global_state, local_rewards, global_reward, current_time, pretrain_epsiodes, pre_fixed_idx, te_period, f_memory, eva)
                
                next_lanes_info, next_maps, next_global_state, local_rewards, global_reward, done = self.env.take_action(actions, current_time, pretrain_epsiodes, eva)
                current_time = self.env.get_current_time()
                if current_time%10==0:
                    print ("     algo", self.para_set.MODEL_NAME)
                    print ('     rewards: eva global', global_reward)
                    # print ("training pv", pv)

            next_maps = []
            lanes_info = next_lanes_info
            maps = next_maps
            global_state = next_global_state 
            last_actions = actions
            runcount+=1
            
        print("END")
        #save the model result
        model_file_name = "model.ckpt"
        model_saver.save(self.sess, os.path.join(self.path_set.PATH_TO_MODEL, model_file_name))
        if self.para_set.MODEL_NAME in ["QMix","VDN"]:
            thepath=os.path.join(self.path_set.PATH_TO_MODEL, "rnn_state")
            np.save(thepath, pv)
            
        #plot 
        plotter = self.Plotter(self)
        plotter.plot_driver()

    def evaluation(self, lanes_info, maps, last_actions, global_state, local_rewards, global_reward, current_time, pretrain_epsiodes, pre_fixed_idx, te_period, f_memory, eva):
        interval = current_time - pre_fixed_idx 
        # if 0.5*te_period<current_time - pre_fixed_idx<te_period: #pure exploration period 
        eva = False 
        actions = self.alg.action_choose(lanes_info, maps, last_actions, global_state, 1, 0, current_time, self.sess) #[n_agents]
        # print ("current_time", current_time)
        # if abs(0.75*te_period - (current_time-pre_fixed_idx))<0.001:
        #     self.cumulative_global_reward = []
        #     self.cumulative_individual_reward = []
        if 0.75*te_period<= (current_time-pre_fixed_idx) <te_period:
            eva = True
            memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(current_time, local_rewards, global_reward)
            f_memory.write(memory_str + "\n")
            f_memory.close()
            # if current_time 
            # print ("     algo", self.para_set.MODEL_NAME)
            # print ('     rewards: eva global', global_reward)
        #     self.cumulative_global_reward.append(global_reward)
        #     self.cumulative_individual_reward.append(local_rewards)
        # if te_period-5 == (current_time-pre_fixed_idx):
        #     report_global_reward = np.average(self.cumulative_global_reward)
        #     report_ind_reward = np.average(np.array(self.cumulative_individual_reward),axis=0)
            # print ("     algo", self.para_set.MODEL_NAME)
            # runtime = pytime.time()-start_time
            # print ('     running time', runtime)
            # print ('     rewards: eva global', report_global_reward)
            # memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(current_time, report_ind_reward, report_global_reward)
            # f_memory.write(memory_str + "\n")
            # f_memory.close()
 
         
        return actions, eva

    def evaluation_rnn(self, lanes_info, maps, last_actions, global_state, local_rewards, global_reward, current_time, pretrain_epsiodes, pre_fixed_idx, te_period, pv, f_memory, eva):
        interval = current_time - pre_fixed_idx 
        # if 0.5*te_period<current_time - pre_fixed_idx<te_period: #pure exploration period 
        eva = False 
        
        actions,pv = self.alg.rnn_choose(lanes_info, maps, last_actions, global_state, pv, 1, 0, current_time, self.sess) #[n_agents]

        # print ("current_time", current_time)
        # if abs(0.75*te_period - (current_time-pre_fixed_idx))<0.001:
        #     self.cumulative_global_reward = []
        #     self.cumulative_individual_reward = []
        if 0.75*te_period<= (current_time-pre_fixed_idx) <te_period:
            eva = True
            memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(current_time, local_rewards, global_reward)
            f_memory.write(memory_str + "\n")
            f_memory.close()
            # if current_time 
            # print ("     algo", self.para_set.MODEL_NAME)
            # print ('     rewards: eva global', global_reward)
        #     self.cumulative_global_reward.append(global_reward)
        #     self.cumulative_individual_reward.append(local_rewards)
        # if te_period-5 == (current_time-pre_fixed_idx):
        #     report_global_reward = np.average(self.cumulative_global_reward)
        #     report_ind_reward = np.average(np.array(self.cumulative_individual_reward),axis=0)
            # print ("     algo", self.para_set.MODEL_NAME)
            # runtime = pytime.time()-start_time
            # print ('     running time', runtime)
            # print ('     rewards: eva global', report_global_reward)
            # memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(current_time, report_ind_reward, report_global_reward)
            # f_memory.write(memory_str + "\n")
            # f_memory.close()
        
        return actions, pv, eva



    def test(self):
        # testing may take longer since it's on more complex environment 
        # all_subdirs = [d for d in os.listdir(self.path_set.PATH_TO_MODEL) if os.path.isdir(d)]
        # latest_subdir = max(all_subdirs, key=os.path.getmtime)
        # pv = self.alg.pv.eval(session=self.sess)
        pv = np.zeros((self.n_agents, 64), dtype='f')
        self.para_set.NOTRAIN = True
        if self.para_set.MODEL_NAME not in ['Random', 'Static']:
            tf.train.Saver().restore(self.sess, os.path.join(self.path_set.PATH_TO_MODEL, "model.ckpt"))
        
        # if self.para_set.MODEL_NAME in ["QMix","VDN"]:
        #     pv = np.load(os.path.join(self.path_set.PATH_TO_MODEL, "rnn_state.npy"))
        
        epsilon_start = 0.0
        epsilon = epsilon_start
        current_time = self.env.get_current_time()  # in seconds
        local_rewards = np.zeros(self.n_agents)
        total_run_cnt = self.para_set.RUN_COUNTS
        file_name_memory = os.path.join(self.path_set.PATH_TO_OUTPUT, "memories.txt")
        pretrain_epsiodes = 1000
        last_actions = np.random.choice([0,1], size=self.n_agents, p=np.ones(2)*0.5)
        eva = True 
        # sys.exit(0)
        # start experiment, there is no explict "done" in TFL
        # pv = np.zeros((self.n_agents,64))
       
        while current_time < total_run_cnt:
            f_memory = open(file_name_memory, "a")
            # get state, which will be used to get the actions
            lanes_info, maps, global_state, done = self.env.get_observation()
            #directly apply the greedy option
            if self.para_set.MODEL_NAME in ['Static']: 
                # actions = np.random.choice([0,1], size=self.n_agents, p=np.ones(2)*0.5)
                actions = np.zeros((self.n_agents,))
            if self.para_set.MODEL_NAME in ['Random']:
                actions = np.random.choice([0,1], size=self.n_agents, p=np.ones(2)*0.5)
             
            if self.para_set.MODEL_NAME in ["IDQN", "QCombo","VDN","QMix", "COMA","IAC"]:##choosing from the regular network
                if self.rnn:##choosing from the rnn frame work
                    actions, pv = self.alg.rnn_choose(lanes_info, maps, last_actions, global_state, pv, 1, epsilon, current_time, self.sess) #[n_agents] 
                else:
                    actions = self.alg.action_choose(lanes_info, maps, last_actions, global_state, 1, epsilon, current_time, self.sess) #[n_agents]
               
                # print ("actions,", actions, pv)
            # if self.para_set.MODEL_NAME in ["COMA","IAC"]:
            #     actions = self.alg.run_actor_target(lanes_info, maps, last_actions, 1, epsilon, self.sess)
            next_lanes_info, next_maps, global_state, local_rewards, global_rewards, done = self.env.take_action(actions, current_time, pretrain_epsiodes, eva)
            # print ('rewards: local', local_rewards)
            # cur_phases = global_state[:self.n_agents]
            # next_phases = global_state[self.n_agents:]
            if current_time >= self.para_set.PRETRAIN_EPSIODES:
                memory_str = 'time = {}\tlocal_reward = {}\tglobal_reward = {}\t'.format(current_time, local_rewards, global_rewards)
                f_memory.write(memory_str + "\n")
                f_memory.close()
            current_time = self.env.get_current_time()
            lanes_info = next_lanes_info
            maps = next_maps  
            last_actions = actions
        print("END")
        #plot 
        shutil.copy(os.path.join(code_dir, "analysis.py"),
                    os.path.join(self.path_set.PATH_TO_OUTPUT, "analysis.py"))
        
        plotter = self.Plotter(self)
        plotter.plot_driver()



def main(memo, algo_choice, train_test, f_prefix, naive_test=False):
    #set the config file 
    # player.train(sumo_cmd_pretrain_str, if_pretrain=True, use_average=True)
    if train_test == "train":
        if algo_choice in ["Static","Random"]:
            player = TrafficLightDQN(memo, algo_choice, f_prefix)
            player.test()
        else: 
            player = TrafficLightDQN(memo, algo_choice, f_prefix)
            player.train()
            # player = TrafficLightDQN(memo, algo_choice, f_prefix, 'test', naive_test, overwrite)
            # player.test()
        
    elif train_test == "test":
        player = TrafficLightDQN(memo, algo_choice, f_prefix)
        player.test()
    else: 
        print ("Clean starts###")
        player = TrafficLightDQN(memo, algo_choice, f_prefix) #clean folder option

import collections 
def nested_dict():
    return collections.defaultdict(nested_dict)


# nn_dic = {'fc':0, 'rnn':1}
# feature_dic = {'others':0,'no_others':1}
four_unequal_seed = {1:22273, 2:59970, 3:25064}
four_unequal_hard_seed = {4:10, 5:678, 6:33908}
four_equal_seed = {7:7650, 8:24580, 9:669, 10:32655, 11:26998}
two_unequal_seed = {16:59551, 12:33458, 13:53635, 14:27650, 15:25725}
six_unequal_seed = {17:2972, 18:11549}
flowname_dict = {1:'unequal',2:'unequal',3:'unequal',
    4:'unequal_more',5:'unequal_more',6:'unequal_more',
    7:'equal',8:'equal',9:'equal',10:'equal', 11:'equal',
    12:'unequal',13:'unequal',14:'unequal',15:'unequal',16:'unequal',
    17:'unequal',18:'unequal'}
all_dict = {}
all_dict.update(two_unequal_seed)
all_dict.update(four_equal_seed)
all_dict.update(four_unequal_hard_seed)
all_dict.update(four_unequal_seed)
all_dict.update(six_unequal_seed)
runtime_seed = all_dict

seed_key_list = [12]
program_type = 'num'
# "QCombo", "COMA","IAC", "QMix", "IDQN", "VDN", 'Random',"Static"
list_model_name = ["IDQN", "QCombo", "COMA", "IAC", "QMix", "VDN",'Random','Static']
list_model_name = ["QCombo"]
search = False   
searchingname = "test_d(-0.5)_q(-0.5)_w(-0.5)_f(-1)_obs(v-s-t-na)"
train_test = 'train'
rnn = False          
include_others = False
include_labels = True  
include_last_actions = False
#equal, unequal, stochastic
subprob = False  
rout = defaultdict(dict)
rout['TRAIN']['X'] = 1
rout['TRAIN']['Y'] = 2
rout['TEST']['X'] = 1
rout['TEST']['Y'] = 2
length = {'long':400, 'short':400, 'inner':400}
# model_for_test = 1
truncated = 100
iteration = 1
PATH_TO_CONF = os.path.join("conf", setting_memo)
dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
if subprob:
    list_model_name = ['QCombo']
# The experiment configure
for the_seed_key in seed_key_list:
    SEED = runtime_seed[the_seed_key]
    dic_exp["SEED"] = SEED
    random.seed(SEED)
    np.random.seed(SEED)
    flowname = flowname_dict[the_seed_key]
    if flowname == 'equal':
        program = 'equal'
    else:
        program = 'unequal'

    if train_test == 'test' and rout['TRAIN']['X'] ==2:
        program = 'stochastic'
    # The algo configure
    # if train_test == 'test':
        # list_model_name = [m for m in list_model_name if m not in ['Random','Static']]
    for model_name in list_model_name:
        dic_algo = json.load(open(os.path.join(PATH_TO_CONF, "deeplight_agent.conf"), "r"))
        dic_algo['STAGE'] = 1
        dic_algo['MODEL_NAME'] = model_name
        dic_algo['TRAIN_TEST'] = train_test
        dic_algo['ALPHA'] = 0.2
        dic_algo['Q_units'] = 256
        dic_algo["BATCH_SIZE"] = dic_exp["BATCH_SIZE"] 
        json.dump(dic_algo, open(os.path.join(PATH_TO_CONF, "{}_agent.conf".format(model_name)), "w"), indent=4)
        dic_exp["MODEL_NAME"] = model_name
        if model_name in ['Static','Random']:
            dic_exp["RUN_COUNTS"] = 5200 #4000
        else:    
            dic_exp["RUN_COUNTS"] = 12500 #This is should at least less or equal to the traffic flow end time 
         #13000
        if train_test == "test":
            dic_exp["RUN_COUNTS"] = 5500 #5500
        dic_exp['FLOWNAME']=flowname 
        dic_exp["TRAIN_TEST"] = train_test
        dic_exp["SUBPROB"] = subprob 
        dic_exp["SEARCHING"]= searchingname

        dic_exp["RNN"] = rnn
        dic_exp['INCLUDE_OTHERS'] = include_others
        dic_exp['INCLUDE_LAST_ACTIONS'] = include_last_actions
        dic_exp['INCLUDE_LABELS'] = include_labels 
        
        plot = {}
        plot['global_reward'] = True
        dic_exp["PLOT"] = plot
        dic_exp["PRETRAIN_EPSIODES"] = 2500
        dic_exp["EPOCHS_PERIOD"] = 5
        dic_exp["TRAIN_EVA_PERIOD"] = 400
        dic_exp["EPOCHS"] = 100
        dic_exp["BATCH_SIZE"] = 30
        dic_exp["PROGRAM"] = program
        json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)
        # change MIN_ACTION_TIME correspondingly, the environment configure
        dic_sumo = json.load(open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "r"))
        # if model_name == "IDQN":
        dic_sumo["MIN_ACTION_TIME"] = 5
        # else:
        # dic_sumo["MIN_ACTION_TIME"] = 5
        reward_term = False 

        dic_sumo['REWARDS_INFO_DICT']['delay'] = [True, -0.5]
        dic_sumo['REWARDS_INFO_DICT']['queue_length'] = [True, -0.5]
        dic_sumo['REWARDS_INFO_DICT']['wait_time'] = [True, -0.5]
        dic_sumo['REWARDS_INFO_DICT']['emergency'] = [True, -0.25]
        dic_sumo['REWARDS_INFO_DICT']['flickering'] = [True, -1]
        dic_sumo['REWARDS_INFO_DICT']['duration'] = [False, 1]
        dic_sumo['REWARDS_INFO_DICT']['partial_duration'] = [False, 1]
        dic_sumo['REWARDS_INFO_DICT']['num_of_vehicles_left'] = [True, 1]
        dic_sumo['REWARDS_INFO_DICT']['duration_of_vehicles_left'] = [True, 1]
        json.dump(dic_sumo, open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "w"), indent=4)


        if program == 'stochastic':  
            flow_x = nested_dict()
            flow_y = nested_dict() 
        else:
            flow_x = defaultdict(dict)
            flow_y = defaultdict(dict)
        # The traffic flow configure
        dic_flow={}
        dic_flow["GUI"] = True 
        dic_flow["PROGRAM"] = program
        dic_flow['PROGRAM_TYPE'] = program_type 
        dic_flow["VEHICLE_SPEED"] = 5


        if dic_exp["TRAIN_TEST"] == 'train':
            dic_flow["Y_ROUTES"] = rout['TRAIN']['Y']
            dic_flow["X_ROUTES"] = rout['TRAIN']['X']
        else:
            dic_flow["Y_ROUTES"] = rout['TEST']['Y']
            dic_flow["X_ROUTES"] = rout['TEST']['X']


        # switch, equal, unequal, synthetic
        if dic_flow['PROGRAM'] == 'equal':
            flow_x["STATUS"] = True
            for i in range(dic_flow["X_ROUTES"]):
                num = 700 
                flow_x[i]["begin"] = 0
                flow_x[i]["end"] = dic_exp["RUN_COUNTS"]
                flow_x[i]["NUM"] = num
                if dic_flow['PROGRAM_TYPE'] == 'prob':
                    flow_x[i]["PROB"] = 0.5
             

            flow_y["STATUS"] = True
            for i in range(dic_flow["Y_ROUTES"]):
                num = 700 
                flow_y[i]["begin"] = 0
                flow_y[i]["end"] = dic_exp["RUN_COUNTS"]
                flow_y[i]["NUM"] = num  
                if dic_flow['PROGRAM_TYPE'] == 'prob':
                    flow_y[i]["PROB"] = 0.5
             


        if dic_flow['PROGRAM'] == 'switch':
            flow_x["STATUS"] = True
            x_num = [500] * dic_flow["X_ROUTES"]
            for i in range(dic_flow["X_ROUTES"]):
                flow_x[i]["begin"] = 0
                flow_x[i]["end"] = dic_exp["RUN_COUNTS"]/2
                flow_x[i]["NUM"] = x_num[i]  

            flow_y["STATUS"] = True
            y_num = [600] * dic_flow["Y_ROUTES"]
            for i in range(dic_flow["Y_ROUTES"]):
                flow_y[i]["begin"] = dic_exp["RUN_COUNTS"]/2
                flow_y[i]["end"] = dic_exp["RUN_COUNTS"]
                flow_y[i]["NUM"] = y_num[i]  

        if dic_flow['PROGRAM'] in ['unequal']:
            flow_x["STATUS"] = True
            for i in range(dic_flow["X_ROUTES"]):
                if i%2 == 0:
                    num = 700 + i*20 #+300
                else:
                    num = 300 - i*20 #+300
                flow_x[i]["begin"] = 0
                flow_x[i]["end"] = dic_exp["RUN_COUNTS"]
                flow_x[i]["NUM"] = num
                if dic_flow['PROGRAM_TYPE'] == 'prob':
                    if i%2 == 0: 
                        flow_x[i]["PROB"] = 0.9
                    else: 
                        flow_x[i]["PROB"] = 0.5
              

            flow_y["STATUS"] = True
            for i in range(dic_flow["Y_ROUTES"]):
                if i%2 == 0:
                    num = 10 + i*20 #+100
                else:
                    num = 600 + i*20 ##+300
                flow_y[i]["begin"] = 0
                flow_y[i]["end"] = dic_exp["RUN_COUNTS"]
                flow_y[i]["NUM"] = num
                if dic_flow['PROGRAM_TYPE'] == 'prob':
                    if i%2 == 0: 
                        flow_y[i]["PROB"] = 0.1
                    else: 
                        flow_y[i]["PROB"] = 0.5
                


        if dic_flow['PROGRAM'] in ['stochastic']:
            flow_x["STATUS"] = True
            flow_x['PHASES'] = 4 # the number of different programs
            theend = int(dic_exp["RUN_COUNTS"])
            thebetween = int(dic_exp["RUN_COUNTS"] - dic_exp["PRETRAIN_EPSIODES"])
            thestart = int(dic_exp["PRETRAIN_EPSIODES"])
            l = range(thestart,theend)
            # l2 = random.sample(list(range(1,dic_exp["RUN_COUNTS"])), flow_x['PHASES']-1)#select k-1 random numbers from 0-run_counts
            l2 = l[0::int(thebetween/3)]
            l2=sorted(l2)
            l2.insert(0,0) #adding start
            l2.append(theend) # adding end 
            for k in range(flow_x['PHASES']):           
                s = l2[k]+1
                if k == 0:
                    s = 0
                e = l2[k+1]
                for i in range(dic_flow["X_ROUTES"]):
                    if k in [0,1]: #the first part 
                        # num = 700
                        if i%4 == 0:
                            num = 700 + i*20  
                        else:
                            num = 300 - i*20 
                     
                    if k == 2:
                        if i%2 == 0: #the first rout
                            num = 1000
                        else:
                            num = 800
                                        
                    if k == 3: # the second part of 1000
                        if i%2 == 0:
                            num = 1400
                        else:
                            num = 1000
                    # num = random.randint(10,1000)
                    flow_x[k][i]["begin"] = s
                    flow_x[k][i]["end"] = e
                    flow_x[k][i]["NUM"] = num  

            flow_y["STATUS"] = True
            flow_y['PHASES'] = 4
            # l = range(0,theend)
            # # l2 = random.sample(list(range(1,dic_exp["RUN_COUNTS"])), flow_x['PHASES']-1)#select k-1 random numbers from 0-run_counts
            # l2 = l[0::int(theend/4)]
            # l2=sorted(l2)
            # # l2.insert(0,0) #adding start
            # l2.append(theend) # adding end 
            for k in range(flow_y['PHASES']):
                s = l2[k]+1
                if k == 0:
                    s = 0 
                e = l2[k+1]
                for i in range(dic_flow["Y_ROUTES"]):
                    if k in [0,1]: #the first part of 1000
                        # num = 700
                        if i%2 == 0:
                            num = 10 + i*20  
                        else:
                            num = 600 + i*20 
                    if k == 2:
                        if i%2 == 0: #the first rout
                            num = 900
                        else:
                            num = 700
                    if k == 3: # the second part of 1000
                        if i%2 == 0:
                            num = 400 
                        else:
                            num = 900
                    # num = random.randint(10,1000)
                    flow_y[k][i]["begin"] = s
                    flow_y[k][i]["end"] = e
                    flow_y[k][i]["NUM"] = num  
            

        dic_flow["FLOW_X"] = flow_x
        dic_flow["FLOW_Y"] = flow_y

        dic_flow["LENGTH"] = length 

        json.dump(dic_flow, open(os.path.join(PATH_TO_CONF, "flow.conf"), "w"), indent=4)

        # folder signature to record which folder to save the result
        
        prefix = "Train_({0},{1})_Test_({2},{3})".format(
        rout['TRAIN']['X'], 
        rout['TRAIN']['Y'],
        rout['TEST']['X'],
        rout['TEST']['Y'])
        
        if search == True:
            prefix = "Train_({0},{1})_Test_({2},{3})_{4}".format(
            rout['TRAIN']['X'], 
            rout['TRAIN']['Y'],
            rout['TEST']['X'],
            rout['TEST']['Y'], searchingname)
        
        if rnn:
            prefix = prefix+"_rnn"
        if include_others:
            prefix = prefix+"_others"
        
        main(memo=setting_memo, algo_choice=model_name, train_test=dic_exp["TRAIN_TEST"], 
        f_prefix=prefix, naive_test=dic_exp["SUBPROB"])
        print ("finished {0}".format(model_name))

    

    





