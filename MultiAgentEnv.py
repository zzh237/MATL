# -*- coding: utf-8 -*-

'''
@author: zhi zhang

Interacting with traffic_light_dqn.py 

1) setting up the sumo MDP environment

2) update state, take action

'''

# from agent import State
from sys import platform
import sys
import os
import math 

import numpy as np
import copy
import shutil
import json
import traci 
import traci.constants as tc
import sumolib

from flow.controllers.routing_controllers import GridRouter
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.core.params import NetParams
from flow.scenarios.grid import SimpleGridScenario
#from flow.scenarios.grid.gen import SimpleGridGenerator
from flow.core.traffic_lights import TrafficLights
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles


###### Please Specify the location of your traci module

if platform == "linux" or platform == "linux2":# this is linux
    os.environ['SUMO_HOME'] = '/usr/local'
    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
	            os.path.join(os.environ["SUMO_HOME"], "tools")
	        )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform == "win32":
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\DLR\\Sumo'

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
elif platform =='darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/Downloads/sumo-0.32.0".format(os.getlogin())

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

else:
    sys.exit("platform error")

yeta = 0.15
tao = 2
constantC = 40.0
carWidth = 3.3
grid_width = 4
direction_lane_dict = {"NSG": [1, 0], "SNG": [1, 0], "EWG": [1, 0], "WEG": [1, 0],
                       "NWG": [0], "WSG": [0], "SEG": [0], "ENG": [0],
                       "NEG": [2], "WNG": [2], "SWG": [2], "ESG": [2]}
direction_list = ["NWG", "WSG", "SEG", "ENG", "NSG", "SNG", "EWG", "WEG", "NEG", "WNG", "SWG", "ESG"]

#min_phase_time = [30, 96, 74]
min_phase_time_7 = [10, 35]
# phases_light_7 = ["WNG_ESG_EWG_WEG_WSG_ENG", "NSG_NEG_SNG_SWG_NWG_SEG"]
# WNG_ESG_EWG_WEG_WSG_ENG = "grrr gGGG grrr gGGG".replace(" ", "")
# NSG_NEG_SNG_SWG_NWG_SEG = "gGGG grrr gGGG grrr".replace(" ", "")
# controlSignal = (WNG_ESG_EWG_WEG_WSG_ENG, NSG_NEG_SNG_SWG_NWG_SEG)

sig0 = "GGGrrrGGGrrr"
sig1 = "rrrGGGrrrGGG"
controlSignal = (sig0, sig1)

class SumoAgent:

    
 
    class PageRank_Matrix:
        "Power iteration with random teleports that addresses Spider trap problem or Dead end problem "
        beta = 0.85
        epsilon = 0.0001
    
        def __init__(self, beta=0.85, epsilon=0.0001):
            self.beta = beta
            self.epsilon = epsilon
    
        def distance(self, v1, v2):
            v = v1 - v2
            v = v * v
            return np.sum(v)
    
        def compute(self, G):
            "G is N*N matrix where if j links to i then G[i][j]==1, else G[i][j]==0"
            N = len(G)
            d = np.zeros(N)
            for i in range(N):
                for j in range(N):
                    if (G[j, i] == 1):
                        d[i] += 1
                if d[i]==0:   # i is dead end, teleport always
                    d[i] = N
    
            r0 = np.zeros(N, dtype=np.float32) + 1.0 / N
            # construct stochastic M
            M = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                if (d[i]==N):  # i is dead end
                    for j in range(N):
                        M[j, i] = 1.0 / d[i]
                else:
                    for j in range(N):
                        if G[j, i] == 1:
                            M[j, i] = 1.0 / d[i]
    
            T = (1.0 - self.beta) * (1.0 / N) * (np.zeros((N, N), dtype=np.float32) + 1.0)
            A = self.beta * M +  T
            while True:
                r1 = np.dot(A, r0)
                dist = self.distance(r1, r0)
                if dist < self.epsilon:
                    break
                else:
                    r0 = r1
    
            return r1

    class ParaSet:
        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                #assign object self with the attribute
                setattr(self, key, value)

    class VehiclesSelf():
        initial_speed = 5.0

        def __init__(self):
            # add what ever you need to maintain
            self.id = None
            self.speed = None
            self.wait_time = None
            self.stop_count = None
            self.enter_time = None
            self.has_read = False
            self.first_stop_time = -1
            self.entering = True

    class Tfl():
        def __init__(self, id):
            self.id = id 
            self.entering_lanes = SumoAgent.listdedup(traci.trafficlight.getControlledLanes(id))
            # for lane in self.entering_lanes: 
            self.dic_vehicles = {}
            # using link to get the leaving lanes
            self.leaving_lanes = []
            for i in self.entering_lanes:
                link = traci.lane.getLinks(i)
                link = [i[0] for i in link]
                self.leaving_lanes = self.leaving_lanes+link      
            self.leaving_lanes = SumoAgent.listdedup(self.leaving_lanes)
            self.current_phase = 0
            self.current_phase_duration = 0
            self.yellow_count = 0
            self.action_duration = 0 
        
    
    def init_tfl_class(self):
        self.tfl_s = []
        traficids = traci.trafficlight.getIDList()
        for id in traficids:
            tfl = self.Tfl(id) 
            self.tfl_s.append(tfl)   

    @staticmethod 
    def listdedup(ls):
        ls = sorted(ls)
        controlledlanes = [ls[0]]
        for i in ls[1:]:
            if i == controlledlanes[-1]:
                continue
            else:
                controlledlanes.append(i) 
        return controlledlanes

                     
    def __init__(self, path_set):

        self.path_set = path_set
        conf_file = os.path.join(self.path_set.PATH_TO_CONF, self.path_set.SUMO_AGENT_CONF)
        dic_paras = json.load(open(conf_file, "r"))
        #return key value pair to initiate the para_set, store the configure into a object
        self.para_set = self.ParaSet(dic_paras)

        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.SUMO_AGENT_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.SUMO_AGENT_CONF))


        ##load the traffic flow parameters 
        conf_file = os.path.join(self.path_set.PATH_TO_CONF, self.path_set.FLOW_CONF)
        dic_paras = json.load(open(conf_file, "r"))
        #return key value pair to initiate the para_set, store the configure into a object
        self.flow_set = self.ParaSet(dic_paras)
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.FLOW_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.FLOW_CONF))
        
        sumo_cmd = self.sumo_conf()

        #start sumo from here, needs to use the new library to replace this file 
        self.start_sumo(sumo_cmd)
        self.init_tfl_class()         
         
        # get all lanes in the environment
        # self.all_lanes = traci.lane.getIDList()
        
        #update each trafic light state, the sefl.tfl_s got updated as well 
        self.all_lanes = []
        for tfl in self.tfl_s: 
            self.update_state(tfl)
            self.update_vehicles_state(tfl)
            self.all_lanes+=tfl.entering_lanes+tfl.leaving_lanes 
        
        self.update_global_state()

        lanes_info, _, global_state, _ = self.get_observation()
        self.self_obs_dim = lanes_info[0].shape[0]
        self.global_state_dim = len(global_state)

        self.n_agents = len(self.tfl_s)
        self.rank = self.cal_tfl_pagerank_weigths()

        self.f_log_rewards = os.path.join(self.path_set.PATH_TO_OUTPUT, "log_rewards.txt")
        if not os.path.exists(self.f_log_rewards):
            f = open(self.f_log_rewards, 'w')
            # valid_reward_keys = list(self.para_set.REWARDS_INFO_DICT.keys())
            valid_reward_keys = [k for k, v in self.para_set.REWARDS_INFO_DICT.items() if v[0]]
            list_reward_keys = np.sort(valid_reward_keys)
            # ['traficlight_id', 'num_of_vehicles_in_trafic_light_range','num_of_vehicles_at_entering'])
            head_str = "itr\ttfl\tcount\taction\tcur\tnext\tduration\t" + '\t'.join(list_reward_keys) + '\n'
            f.write(head_str)
            f.close()

    def gen_edges(self, row_num, col_num):
        edges = []
        for i in range(col_num):
            edges += ["left" + str(row_num) + '_' + str(i)]
            edges += ["right" + '0' + '_' + str(i)]

        # build the left and then the right edges
        for i in range(row_num):
            edges += ["bot" + str(i) + '_' + '0']
            edges += ["top" + str(i) + '_' + str(col_num)]

        return edges

    def generate_scenario(self, render=None): 
        v_enter = 16.67
        inner_length = self.flow_set.LENGTH['inner']
        long_length = self.flow_set.LENGTH['long']
        short_length = self.flow_set.LENGTH['short']

        n = self.flow_set.Y_ROUTES #routes parallel to the y
        m = self.flow_set.X_ROUTES #routes parallel to the x

        self.area_x = (self.flow_set.Y_ROUTES - 1) * inner_length + 2 * long_length  
        self.area_y= (self.flow_set.X_ROUTES - 1) * inner_length + 2 * long_length  

        num_cars_left = 1 #the number of vehicles put on the left lanes
        num_cars_right = 1
        num_cars_top = 1
        num_cars_bot = 1
        tot_cars = (num_cars_left + num_cars_right) * m \
            + (num_cars_top + num_cars_bot) * n

        grid_array = {
            "short_length": short_length,
            "inner_length": inner_length,
            "long_length": long_length,
            "row_num": m,
            "col_num": n,
            "cars_left": num_cars_left,
            "cars_right": num_cars_right,
            "cars_top": num_cars_top,
            "cars_bot": num_cars_bot
        }

        sumo_params = SumoParams(sim_step=0.1, render=True)

        if render is not None:
            sumo_params.render = render

        vehicles = Vehicles()
        vehicles.add(
            veh_id="human",
            sumo_car_following_params=SumoCarFollowingParams(
                min_gap=2, tau=1.1, max_speed=v_enter),
            routing_controller=(GridRouter, {}),
            num_vehicles=tot_cars)  #tot cars are the current cars in the grid
        
        #the initial phases, doesn't matter will be updated based on the action
        tl_logic = TrafficLights(baseline=False)
        phases = [{
            "duration": "30",
            "minDur": "8",
            "maxDur": "45",
            "state": "GGGrrrGGGrrr"  #clockwise traffic light phases, represents North, east, south, west
        }, {
            "duration": "3",
            "minDur": "3",
            "maxDur": "6",
            "state": "yyyrrryyyrrr"
        }, {
            "duration": "30",
            "minDur": "8",
            "maxDur": "45",
            "state": "rrrGGGrrrGGG"
        }, {
            "duration": "3",
            "minDur": "3",
            "maxDur": "6",
            "state": "rrryyyrrryyy"
        }]
        num_tfls = n*m
        # adding the traffic lights 
        for i in range(num_tfls):
            tl_logic.add("center{}".format(i), tls_type="static", phases=phases, programID=1)


        env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

        additional_net_params = {
            "grid_array": grid_array,
            "speed_limit": 35,
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
            "time-to-teleport": 3000
        }
        initial_config = InitialConfig(spacing="uniform", lanes_distribution=float("inf"), shuffle=True)
        inflow = InFlows()   
        # These edges are the emmission edges for the vechiles, not all the edges, use these 
        # to control if the vechiles should be balanced or not   
        edges_y = []
        for i in range(n): #for every routs parallel to the y 
            edges_y += ["left" + str(m) + '_' + str(i)] # m menas the very top grid
            edges_y += ["right" + '0' + '_' + str(i)] #0 means the very bot grid 

        # build the left and then the right edges
        # build the bot and then the top edges 
        edges_x = []
        for i in range(m): #for every routs parallel to the x, so total m routs  
            edges_x += ["bot" + str(i) + '_' + '0'] # the 0 means the very left grid
            edges_x += ["top" + str(i) + '_' + str(n)] # n means the righmost grid
        
        # if self.flow_set.FLOW_X["STATUS"]:
        if self.flow_set.PROGRAM == 'stochastic':
            for j in range(self.flow_set.FLOW_X['PHASES']):
                index = str(math.floor(j))
                for i in range(len(edges_x)):
                    key = str(math.floor(i/2))
                    inflow.add(
                        veh_type="human",
                        edge=edges_x[i],  #control the bot 0, 1, ..., m edges 
                        begin=self.flow_set.FLOW_X[index][key]['begin'],
                        end=self.flow_set.FLOW_X[index][key]['end'],
                        vehs_per_hour=self.flow_set.FLOW_X[index][key]["NUM"],
                        departLane="free",
                        departSpeed=v_enter) 
                
                for i in range(len(edges_y)):
                    key = str(math.floor(i/2))
                    inflow.add(
                        veh_type="human",
                        edge=edges_y[i],  #control the bot 0, 1, ..., m edges 
                        begin=self.flow_set.FLOW_Y[index][key]['begin'],
                        end=self.flow_set.FLOW_Y[index][key]['end'],
                        vehs_per_hour=self.flow_set.FLOW_Y[index][key]["NUM"],
                        departLane="free",
                        departSpeed=v_enter) 

        else:
            for i in range(len(edges_x)):
                key = str(math.floor(i/2))
                if self.flow_set.PROGRAM_TYPE == 'prob':
                    inflow.add(
                    veh_type="human",
                    edge=edges_x[i],  #control the bot 0, 1, ..., m edges 
                    begin=self.flow_set.FLOW_X[key]['begin'],
                    end=self.flow_set.FLOW_X[key]['end'],
                    probability=self.flow_set.FLOW_X[key]["PROB"],
                    departLane="free",
                    departSpeed=v_enter)
                else:
                    inflow.add(
                        veh_type="human",
                        edge=edges_x[i],  #control the bot 0, 1, ..., m edges 
                        begin=self.flow_set.FLOW_X[key]['begin'],
                        end=self.flow_set.FLOW_X[key]['end'],
                        vehs_per_hour=self.flow_set.FLOW_X[key]["NUM"],
                        departLane="free",
                        departSpeed=v_enter)

            for i in range(len(edges_y)):
                key = str(math.floor(i/2))
                if self.flow_set.PROGRAM_TYPE == 'prob':
                    inflow.add(
                        veh_type="human",
                        edge=edges_y[i], # control the left 0, 1, ... , n edges 
                        begin=self.flow_set.FLOW_Y[key]['begin'],
                        end=self.flow_set.FLOW_Y[key]['end'],
                        probability=self.flow_set.FLOW_Y[key]["PROB"],
                        departLane="free",
                        departSpeed=v_enter)
                else:
                    inflow.add(
                        veh_type="human",
                        edge=edges_y[i], # control the left 0, 1, ... , n edges 
                        begin=self.flow_set.FLOW_Y[key]['begin'],
                        end=self.flow_set.FLOW_Y[key]['end'],
                        vehs_per_hour=self.flow_set.FLOW_Y[key]["NUM"],
                        departLane="free",
                        departSpeed=v_enter)
                #now to add another inflow which follow the prior inflow 
        # if self.flow_set.FLOW_Y["STATUS"]:   
            # if self.flow_set.PROGRAM == 'stochastic':
            #     for j in range(self.flow_set.FLOW_Y['PHASES']):
            #         index = str(math.floor(j/2))
            #         for i in range(len(edges_y)):
            #             key = str(math.floor(i/2))
            #             inflow.add(
            #                 veh_type="human",
            #                 edge=edges_y[i],  #control the bot 0, 1, ..., m edges 
            #                 begin=self.flow_set.FLOW_Y[index][key]['begin'],
            #                 end=self.flow_set.FLOW_Y[index][key]['end'],
            #                 vehs_per_hour=self.flow_set.FLOW_Y[index][key]["NUM"],
            #                 departLane="free",
            #                 departSpeed=v_enter) 
            # else:
            #     for i in range(len(edges_y)):
            #         key = str(math.floor(i/2))
            #         inflow.add(
            #             veh_type="human",
            #             edge=edges_y[i], # control the left 0, 1, ... , n edges 
            #             begin=self.flow_set.FLOW_Y[key]['begin'],
            #             end=self.flow_set.FLOW_Y[key]['end'],
            #             vehs_per_hour=self.flow_set.FLOW_Y[key]["NUM"],
            #             departLane="free",
            #             departSpeed=v_enter)
        

        net_params = NetParams(
            inflows=inflow,
            no_internal_links=False,
            additional_params=additional_net_params)

        net_params = NetParams(inflows=inflow, 
            no_internal_links=False, additional_params=additional_net_params)

        scenario = SimpleGridScenario(
            name="grid-intersection",
 #           generator_class=SimpleGridGenerator,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=tl_logic)
        return scenario 
    

    def delete_file(self):
        test = os.listdir(self.path_set.PATH_TO_OUTPUT)
        for item in test:
            if item.endswith(".xml"):
                os.remove(os.path.join(self.path_set.PATH_TO_OUTPUT, item))
    
    def sumo_conf(self):
        #need the sumo configure to control this
        print ("sumo", os.environ['SUMO_HOME'])
        sumo_path = os.environ['SUMO_HOME']
        if self.flow_set.GUI == True: 
            sumocfg = self.generate_scenario(True).cfg
            thecfgplace = os.path.split(sumocfg)[0] 
            thecfgfile = os.path.split(sumocfg)[-1]
            thecfgfile = thecfgfile[:-9]
            theroutfile = thecfgfile+'.rou.xml'
            oldfile = os.path.join(thecfgplace, theroutfile)
            newfile = os.path.join(self.path_set.PATH_TO_OUTPUT, theroutfile)
            self.delete_file()
            shutil.copy(oldfile, newfile)
            
            sumoBinary = os.path.join(sumo_path, "bin/sumo-gui")
        else: 
            sumocfg = self.generate_scenario(None).cfg
            thecfgplace = os.path.split(sumocfg)[0] 
            thecfgfile = os.path.split(sumocfg)[-1]
            thecfgfile = thecfgfile[:-9]
            theroutfile = thecfgfile+'.rou.xml'
            oldfile = os.path.join(thecfgplace, theroutfile)
            newfile = os.path.join(self.path_set.PATH_TO_OUTPUT, theroutfile)
            self.delete_file()
            shutil.copy(oldfile, newfile)
            sumoBinary = os.path.join(sumo_path, "bin/sumo")
        sumoCmd = [sumoBinary, '-c', sumocfg]
        return sumoCmd

    def start_sumo(self, sumo_cmd_str):
        traci.start(sumo_cmd_str)
        for i in range(20):
            traci.simulationStep()

    def end_sumo(self):
        traci.close()

    def get_current_time(self):
        return traci.simulation.getCurrentTime() / 1000

    def load_conf(self, conf_file):
        class ParaSet:
            def __init__(self, dic_paras):
                for key, value in dic_paras.items():
                    setattr(self, key, value)
        dic_paras = json.load(open(conf_file, "r"))   
        return ParaSet(dic_paras)

   

    class State(object):
        # D_QUEUE_LENGTH = (12,)
        # D_NUM_OF_VEHICLES = (12,)
        # D_WAITING_TIME = (12,)
        # D_MAP_FEATURE = (150,150,1,)
        # D_CUR_PHASE = (1,)
        # D_NEXT_PHASE = (1,)
        # D_TIME_THIS_PHASE = (1,)
        # D_IF_TERMINAL = (1,)
        # D_HISTORICAL_TRAFFIC = (6,)

        # ==========================

        def __init__(self,
                    queue_length, num_of_vehicles, waiting_time, delay, speed,  
                    map_feature,
                    cur_phase,
                    next_phase,
                    time_this_phase,
                    if_terminal):

            self.queue_length = queue_length
            self.num_of_vehicles = num_of_vehicles
            self.waiting_time = waiting_time
            self.delay = delay 
            self.speed = speed
            self.map_feature = map_feature
            self.cur_phase = cur_phase
            self.next_phase = next_phase
            self.time_this_phase = time_this_phase
            self.if_terminal = if_terminal
            self.historical_traffic = None
            self.observation = np.array([self.queue_length, self.num_of_vehicles, self.waiting_time, 
                                    self.cur_phase, self.next_phase])


    def getMapOfVehicles(self):
        '''
        get the vehicle positions as NIPS paper
        :param area_y:
        :return: numpy narray
        '''

        length_num_grids = int(self.area_y/ grid_width)
        width_num_grids = int(self.area_x / grid_width)
        mapOfCars = np.zeros((length_num_grids, width_num_grids))

        vehicle_id_list = traci.vehicle.getIDList()
        for vehicle_id in vehicle_id_list:
            vehicle_position = traci.vehicle.getPosition(vehicle_id)  # (double,double),tuple

            transform_tuple = self.vehicle_location_mapper(vehicle_position)  # call the function
            mapOfCars[transform_tuple[0], transform_tuple[1]] = 1
        return mapOfCars

    
    def vehicle_location_mapper(self, coordinate):
        transformX = math.floor(coordinate[0] / grid_width)
        transformY = math.floor((self.area_y- coordinate[1]) / grid_width)
        length_num_grids = int(self.area_y/grid_width)
        width_num_grids = int(self.area_x/grid_width)
        transformY = length_num_grids-1 if transformY == length_num_grids else transformY
        transformX = width_num_grids-1 if transformX == width_num_grids else transformX
        tempTransformTuple = (transformY, transformX) #this is corresponding to the image axis, 
        return tempTransformTuple


    def status_calculator(self, listLanes):
        laneQueueTracker=[]
        laneNumVehiclesTracker=[]
        laneWaitingTracker=[]
        laneDelayTracker = []
        laneSpeedTracker = []
        #================= COUNT HALTED VEHICLES (I.E. QUEUE SIZE) (12 elements)
        for lane in listLanes:
            laneQueueTracker.append(traci.lane.getLastStepHaltingNumber(lane))

        # ================ count vehicles in lane
        for lane in listLanes:
            laneNumVehiclesTracker.append(traci.lane.getLastStepVehicleNumber(lane))

        # ================ cum waiting time in minutes
        for lane in listLanes:
            laneWaitingTracker.append(traci.lane.getWaitingTime(str(lane)) / 60)

        for lane in listLanes:
            laneDelayTracker.append(1 - traci.lane.getLastStepMeanSpeed(str(lane)) / traci.lane.getMaxSpeed(str(lane)))
        
        for lane in listLanes:
            laneSpeedTracker.append(traci.lane.getLastStepMeanSpeed(str(lane)))
        
        return [laneQueueTracker, laneNumVehiclesTracker, laneWaitingTracker, laneDelayTracker, laneSpeedTracker]

    def update_state(self, tfl):
        # The basic elements of the state 
        # status_tracker = [laneQueueTracker, laneNumVehiclesTracker, laneWaitingTracker, mapOfCars]
        status_tracker = self.status_calculator(tfl.entering_lanes) #only the incomming lanes matter
        # ================ get position matrix of vehicles on lanes
        # mapOfCars = self.getMapOfVehicles()
        mapOfCars = np.array(self.getMapOfVehicles())
        # newshape_x = (self.flow_set.X_ROUTES+1)*100 
        # newshape_y = (self.flow_set.Y_ROUTES+1)*100 
        nlanes = len(tfl.entering_lanes)
        tfl.state = self.State(
            queue_length=status_tracker[0],
            num_of_vehicles=status_tracker[1],
            waiting_time=status_tracker[2],
            delay = status_tracker[3],
            speed = status_tracker[4],
            # map_feature=np.reshape(np.array(mapOfCars), newshape=(newshape_x, newshape_y)),
            map_feature =mapOfCars,
            cur_phase=tfl.current_phase,
            next_phase=(tfl.current_phase + 1) % len(self.para_set.MIN_PHASE_TIME),
            time_this_phase=tfl.current_phase_duration,
            if_terminal=False
        )

    def update_global_state(self):
        # The basic elements of the state 
        # status_tracker = [laneQueueTracker, laneNumVehiclesTracker, laneWaitingTracker, mapOfCars]
        status_tracker = self.status_calculator(self.all_lanes) #only the incomming lanes matter
        # ================ get position matrix of vehicles on lanes
        mapOfCars = np.array(self.getMapOfVehicles())
        # newshape_x = (self.flow_set.X_ROUTES+1)*100 
        # newshape_y = (self.flow_set.Y_ROUTES+1)*100 
        self.state = self.State(
            queue_length=status_tracker[0],
            num_of_vehicles=status_tracker[1],
            waiting_time=status_tracker[2],
            delay = status_tracker[3],
            speed = status_tracker[4], 
            # map_feature=np.reshape(np.array(mapOfCars), newshape=(newshape_x, newshape_y)),
            map_feature = mapOfCars,
            cur_phase=None,
            next_phase=None,
            time_this_phase=None,
            if_terminal=False
        )
    
    
    def get_observation(self):
        # This function will be used to construct all the relevant states and rewards 
        lanes_info = []
        self_state = []
        maps = []
        current_phases = []
        next_phases = []
        
        # global_state = self.state.queue_length + self.state.num_of_vehicles+self.state.waiting_time + self.state.delay + self.state.speed
        global_state = self.state.num_of_vehicles+self.state.speed+self.state.waiting_time
        
        done = False
        for tfl in self.tfl_s:
            #lanes_info is a matrix that row*col = lanes*features, sum up routes 
            q = [tfl.state.queue_length[0]+tfl.state.queue_length[3], tfl.state.queue_length[1]+tfl.state.queue_length[2]] 
            v = [tfl.state.num_of_vehicles[0]+tfl.state.num_of_vehicles[3], tfl.state.num_of_vehicles[1]+tfl.state.num_of_vehicles[2]] 
            t = [tfl.state.waiting_time[0]+tfl.state.waiting_time[3], tfl.state.waiting_time[1]+tfl.state.waiting_time[2]] 
            d = [tfl.state.delay[0]+tfl.state.delay[3], tfl.state.delay[1]+tfl.state.delay[2]] 
            s = [tfl.state.speed[0]+tfl.state.speed[3], tfl.state.speed[1]+tfl.state.speed[2]] 
            c = tfl.state.cur_phase
            n = tfl.state.next_phase
            u = tfl.state.time_this_phase
            # lanes_info.append(np.stack((q,v,t), axis=1))
            maps.append(tfl.state.map_feature)
            #self_state is a vector with three values 
            # obs = q+v+t+[tfl.state.cur_phase, tfl.state.next_phase, tfl.state.time_this_phase]
            obs = v+s+t
            lanes_info.append(np.array(obs)) 
            done = done if tfl.state.if_terminal==False else True
            # current_phases.append(tfl.state.cur_phase), global state is a very long vector size = n_agents * all_features
            # global_state += q+v+t+[tfl.state.cur_phase, tfl.state.next_phase, tfl.state.time_this_phase] 
            # global_state += [c,n,u]    
            # next_phases.append(tfl.state.next_phase)
        # global_state = current_phases + next_phases
        return lanes_info, maps, global_state, done  

        
    def get_tfl_ids(self):
        return [tfl.id for tfl in self.tfl_s]

    def take_action(self, action_pred, iter_num, pretrain_epsiodes, eva):
        #action 0: not change the current duration, 1: change the current duration 
        action = [0]*self.n_agents #[n_agents]
        local_reward = [0]*self.n_agents #[n_agents]
        lanes_info = [] #[n_agents, 2*3]
        self_state = [] #[n_agents, 3]
        maps = []
        done = False
        current_time = self.get_current_time()
        for tfl in self.tfl_s: 
            tfl.rewards_detail_dict_list = []

        for i in range(self.para_set.MIN_ACTION_TIME):        
            for index, tfl in enumerate(self.tfl_s): 
                # v = self.update_vehicles_state(tfl, tfl.dic_vehicles)
                # tfl.dic_vehicles = v 

                # if (tfl.current_phase_duration < self.para_set.MIN_PHASE_TIME[current_phase_number]): #[0,0] current the min_phase_time is zero for both phases
                #     action[index] = 0 
                #scalar, the current action time is 5
                tfl.action_in_second = 0 
                if 1<= tfl.action_duration <= 5: 
                    tfl.action_in_second = 0  
                
                if action_pred[index] == 1 and tfl.action_duration == 0: #prevent action change within 5 seconds, only the first second is 1 
                    tfl.action_in_second = 1
                

                
                if tfl.action_in_second == 0:
                    next_phase = tfl.current_phase
                    next_phase_duration = tfl.current_phase_duration
                               
                 
                if tfl.action_in_second == 1:                
                
                    # if tfl.yellow_count < 3:
                    #     #before the 1, three yellow steps 
                    #     # self.set_yellow(tfl, vehicle_dict,rewards_info_dict,f_log_rewards, rewards_detail_dict_list, iter_num)
                    #     # set_all_red(vehicle_dict,rewards_info_dict,f_log_rewards, node_id=node_id)
                    #     traci.trafficlights.setRedYellowGreenState(node_id, Yellow)
                    #     tfl.yellow_count+=1
                    #     next_phase_duration = 0 
                    # else: 
                    denomi = len(controlSignal)
                    nomi = tfl.current_phase + 1
                     # (0+1)%2 = 1, (1+1)%2 = 0, which can change the signal 
                    next_phase = (tfl.current_phase + 1) % len(controlSignal)
                    next_phase_time_eclipsed = 0
                    traci.trafficlights.setRedYellowGreenState(tfl.id, controlSignal[next_phase])
                    next_phase_duration = 0
                
                # if tfl.yellow_count == 3: 
                #     tfl.yellow_count = 0 
                #     tf.action_duration = 0
                tfl.action_duration +=1
                if tfl.action_duration == 5:
                    tfl.action_duration = 0 
                
                tfl.current_phase = next_phase
                tfl.current_phase_duration = next_phase_duration+1     
                tfl.vehicle_dict = self.update_vehicles_state(tfl)

            timestamp = traci.simulation.getCurrentTime() / 1000
            traci.simulationStep()
            #calculate the rewards of all the vehicles and also update the vehicles state for every second 
            # during the 5 seconds, then later accumulate the rewards so that to get the final reward 
            for index, tfl in enumerate(self.tfl_s): 
                self.log_rewards(tfl, tfl.action_in_second, timestamp, iter_num, eva)
                self.update_vehicles_state(tfl)

        for index, tfl in enumerate(self.tfl_s):
            #adding the scalling 100 to prevent from zeros , the cummulative rewardd
            local_reward[index] = self.cal_reward_from_list(tfl)
            self.update_state(tfl)
        self.update_global_state()
            
        global_reward = self.cal_global_reward(local_reward)
        lanes_info, maps, global_state, done = self.get_observation()
        return lanes_info, maps, global_state, local_reward, global_reward, done   

    def log_rewards(self, tfl, action, timestamp, iter_num, eva):
        reward, reward_detail_dict = self.get_rewards_from_sumo(tfl, action)
        list_reward_keys = np.sort(list(reward_detail_dict.keys()))
        valid_reward_keys = [k for k, v in reward_detail_dict.items() if v[0]]
        valid_reward_keys = np.sort(valid_reward_keys)
        reward_str = "{0}\t{1}\t{2}\t{3}".format(iter_num, tfl.id, timestamp, action)
        reward_str = reward_str + "\t{0}\t{1}\t{2}".format(tfl.current_phase, (tfl.current_phase + 1) % len(self.para_set.MIN_PHASE_TIME), tfl.current_phase_duration)
        for reward_key in valid_reward_keys:
            reward_str = reward_str + "\t{0}".format(reward_detail_dict[reward_key][2])
        reward_str += '\n'
        if eva:
            fp = open(self.f_log_rewards, "a")
            fp.write(reward_str)
            fp.close()
        tfl.rewards_detail_dict_list.append(reward_detail_dict)

    def cal_global_reward(self, local_reward):
        # rank = self.cal_tfl_pagerank_weigths()
        # tot = np.zeros((self.n_agents, 3))
        # for i, tfl in enumerate(self.tfl_s):
        #     lanes = tfl.entering_lanes + tfl.leaving_lanes
        #     q,v,t = self.status_calculator(lanes)
        #     sq, sv, st = np.sum(q), np.sum(v), np.sum(t)
        #     tot[i, :] =np.array([sq, sv, st])
        # sums = np.sum(tot,axis=0)
        # pct = tot
        # for j in range(sums.shape[0]):
        #     if sums[j] == 0:
        #         pct[:, j] = 1/self.n_agents
        #     else: 
        #         pct[:, j] = tot[:, j]/sums[j]
        # wgt = np.mean(pct, axis=1)

        # g_reward = np.dot(wgt, local_reward)
        g_reward = np.dot(self.rank, local_reward) 
        return g_reward 

    def cal_tfl_pagerank_weigths(self):
        res = np.zeros((self.n_agents,self.n_agents))
        for i, tfl1 in enumerate(self.tfl_s): 
            for j, tfl2 in enumerate(self.tfl_s):
                if tfl1.id == tfl2.id: 
                    res[i,j] = 1 
                else:
                    outlinks = tfl1.leaving_lanes
                    inlinks = tfl2.entering_lanes
                    flag = 0
                    for k in outlinks:
                        if flag == 1:
                            break 
                        for m in inlinks:
                            if k==m: 
                                flag = 1 
                                break                 
                    res[i,j] = flag 

        PRM = self.PageRank_Matrix(0.85, 0.0001)
        rank = PRM.compute(res)
        return rank 
             

    

    def cal_reward_from_list(self, tfl):
        reward = 0
        rewards_detail_dict_list = tfl.rewards_detail_dict_list
        for i in range(len(rewards_detail_dict_list)): #the 5 seconds
            for k, v in rewards_detail_dict_list[i].items():
                if v[0]:  # True or False
                    reward += v[1] * v[2]
        reward = self.restrict_reward(reward)
        return reward*(1-0.8)

    def get_rewards_from_dict_list(self, rewards_detail_dict_list):
        reward = 0
        for i in range(len(rewards_detail_dict_list)):
            for k, v in rewards_detail_dict_list[i].items():
                if v[0]:  # True or False
                    reward += v[1] * v[2]
        reward = self.restrict_reward(reward)
        return reward

    def restrict_reward(self, reward,func="unstrict"):
        if func == "linear":
            bound = -50
            reward = 0 if reward < bound else (reward/(-bound) + 1)
        elif func == "neg_log":
            reward = math.log(-reward+1)
        else:
            pass
        return reward


    def update_vehicles_state(self, tfl):
        #retrive all online vehicles 
        # vehicle_id_list = traci.vehicle.getIDList()
        dic_vehicles = tfl.dic_vehicles
        vehicle_id_entering_list,vehicle_id_leaving_list = self.get_vehicle_id_entering_leaving(tfl)
        vehicle_id_list = vehicle_id_entering_list+vehicle_id_leaving_list
        for vehicle_id in (set(dic_vehicles.keys())-set(vehicle_id_list)):
            del(dic_vehicles[vehicle_id])# match only to the vehicle id list vehicles

        for vehicle_id in vehicle_id_list:
            # if vehicle_id == 'human_36':
            #     print ('zz')
            if (vehicle_id in dic_vehicles.keys()) == False:  #the new vehicles added 
                vehicle = self.VehiclesSelf()
                vehicle.id = vehicle_id
                traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
                vehicle.speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
                current_sumo_time = traci.simulation.getCurrentTime()/1000 #from ms to s 
                vehicle.enter_time = current_sumo_time 
                # if it enters and stops at the very first
                if (vehicle.speed < 0.1) and (vehicle.first_stop_time == -1):
                    vehicle.first_stop_time = current_sumo_time
                dic_vehicles[vehicle_id] = vehicle
                if (vehicle_id in vehicle_id_entering_list) == False:
                    dic_vehicles[vehicle_id].entering = False
            else:
                dic_vehicles[vehicle_id].speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
                if (dic_vehicles[vehicle_id].speed < 0.1) and (dic_vehicles[vehicle_id].first_stop_time == -1):
                    dic_vehicles[vehicle_id].first_stop_time = traci.simulation.getCurrentTime()/1000
                if (vehicle_id in vehicle_id_entering_list) == False:
                    dic_vehicles[vehicle_id].entering = False

        tfl.dic_vehicles = dic_vehicles 

        

    def set_yellow(self, tfl, dic_vehicles,rewards_info_dict,f_log_rewards,rewards_detail_dict_list,iter_num):
        Yellow = "yyyyyyyyyyyy"
        node_id = tfl.id
        for i in range(3): #After action 1, takes 3 more steps to update the reward, that's why three 0 before the 1  
            timestamp = traci.simulation.getCurrentTime() / 1000
            traci.trafficlights.setRedYellowGreenState(node_id, Yellow)
            traci.simulationStep()
            self.log_rewards(tfl, dic_vehicles, 0, rewards_info_dict, f_log_rewards, timestamp, rewards_detail_dict_list, iter_num)
            self.update_vehicles_state(tfl)
    
    
    def get_vehicle_id_entering_leaving(self, tfl):
        vehicle_id_entering = []
        vehicle_id_leaving = []
        for lane in tfl.entering_lanes:
            vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))
        for lane in tfl.leaving_lanes:
            vehicle_id_leaving.extend(traci.lane.getLastStepVehicleIDs(lane))
        return vehicle_id_entering, vehicle_id_leaving 

    def get_vehicle_id_leaving(self, tfl, vehicle_dict): #total vehicles in the vehicle_dict
        vehicle_id_leaving = []
        vehicle_id_entering, vehicle_id_passing = self.get_vehicle_id_entering_leaving(tfl)
        for vehicle_id in vehicle_dict.keys():
            if not(vehicle_id in vehicle_id_entering) and vehicle_dict[vehicle_id].entering:
                vehicle_id_leaving.append(vehicle_id)
        return vehicle_id_leaving

    def get_rewards_from_sumo(self, tfl, action,):
        reward = 0
        vehicle_dict = tfl.dic_vehicles
        listLanes = tfl.entering_lanes
        reward_detail_dict = copy.deepcopy(self.para_set.REWARDS_INFO_DICT) # This is basically the sumo agent configure files
        vehicle_id_entering_list, vehicle_id_passing = self.get_vehicle_id_entering_leaving(tfl)
        reward_detail_dict['queue_length'].append(self.get_overall_queue_length(listLanes))
        reward_detail_dict['wait_time'].append(self.get_overall_waiting_time(listLanes))
        reward_detail_dict['delay'].append(self.get_overall_delay(listLanes))
        reward_detail_dict['emergency'].append(self.get_num_of_emergency_stops(tfl, vehicle_dict))
        reward_detail_dict['duration'].append(self.get_travel_time_duration(vehicle_dict, vehicle_id_entering_list))
        reward_detail_dict['flickering'].append(self.get_flickering(action))
        reward_detail_dict['partial_duration'].append(self.get_partial_travel_time_duration(vehicle_dict, vehicle_id_entering_list))
        vehicle_id_list = traci.vehicle.getIDList()
        reward_detail_dict['num_of_vehicles_in_system'] = [False, 0, len(vehicle_id_list)]
        reward_detail_dict['num_of_vehicles_at_entering'] = [False, 0, len(vehicle_id_entering_list)]
        vehicle_id_leaving = self.get_vehicle_id_leaving(tfl, vehicle_dict)
        reward_detail_dict['num_of_vehicles_left'].append(len(vehicle_id_leaving))
        reward_detail_dict['duration_of_vehicles_left'].append(self.get_travel_time_duration(vehicle_dict, vehicle_id_leaving))
        for k, v in reward_detail_dict.items():
            if v[0]:  # True or False
                reward += v[1]*v[2]
        reward = self.restrict_reward(reward)#,func="linear")
        return reward, reward_detail_dict

    def get_overall_queue_length(self, listLanes):
        overall_queue_length = 0
        for lane in listLanes:
            overall_queue_length += traci.lane.getLastStepHaltingNumber(lane)
            # print ("queue_length", overall_queue_length)
        return overall_queue_length

    def get_overall_waiting_time(self, listLanes):
        overall_waiting_time = 0
        for lane in listLanes:
            overall_waiting_time += traci.lane.getWaitingTime(str(lane)) / 60.0
        return overall_waiting_time

    def get_overall_delay(self, listLanes):
        overall_delay = 0
        for lane in listLanes:
            overall_delay += 1 - traci.lane.getLastStepMeanSpeed(str(lane)) / traci.lane.getMaxSpeed(str(lane))
        return overall_delay

    # calculate number of emergency stops by vehicle
    def get_num_of_emergency_stops(self, tfl, vehicle_dict):
        emergency_stops = 0
        vehicle_id_entering_list,vehicle_id_leaving_list = self.get_vehicle_id_entering_leaving(tfl)
        vehicle_id_list = vehicle_id_entering_list+vehicle_id_leaving_list
        for vehicle_id in vehicle_id_list:
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
            current_speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
            if (vehicle_id in vehicle_dict.keys()):
                vehicle_former_state = vehicle_dict[vehicle_id]
                if current_speed - vehicle_former_state.speed < -4.5:
                    emergency_stops += 1
            else:
                # print("##New car coming")
                if current_speed - self.VehiclesSelf.initial_speed < -4.5:
                    emergency_stops += 1
        if len(vehicle_dict) > 0:
            return emergency_stops/len(vehicle_dict)
        else:
            return 0

    def get_travel_time_duration(self, vehicle_dict, vehicle_id_list):
        #vehicle_id_list: the leaving vehicles
        travel_time_duration = 0
        for vehicle_id in vehicle_id_list:
            if (vehicle_id in vehicle_dict.keys()):
                travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - vehicle_dict[vehicle_id].enter_time)/60.0
        if len(vehicle_id_list) > 0:
            return travel_time_duration#/len(vehicle_id_list)
        else:
            return 0
    
    def get_flickering(self, action):
        return action

    def get_partial_travel_time_duration(self, vehicle_dict, vehicle_id_list):
        travel_time_duration = 0
        for vehicle_id in vehicle_id_list:
            if (vehicle_id in vehicle_dict.keys()) and (vehicle_dict[vehicle_id].first_stop_time != -1):
                travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - vehicle_dict[vehicle_id].first_stop_time)/60.0
        if len(vehicle_id_list) > 0:
            return travel_time_duration#/len(vehicle_id_list)
        else:
            return 0

    def changeTrafficLight_7(self, tfl, current_phase=0):  # [WNG_ESG_WSG_ENG_NWG_SEG]
        # phases=["WNG_ESG_WSG_ENG_NWG_SEG","EWG_WEG_WSG_ENG_NWG_SEG","NSG_NEG_SNG_SWG_WSG_ENG_NWG_SEG"]
        denomi = len(controlSignal)
        nomi = current_phase + 1 # (0+1)%2 = 1, (1+1)%2 = 0, which can change the signal 
        next_phase = (current_phase + 1) % len(controlSignal)
        next_phase_time_eclipsed = 0
        traci.trafficlights.setRedYellowGreenState(tfl.id, controlSignal[next_phase])
        return next_phase, next_phase_time_eclipsed


    def cal_reward(self, tfl, action):
        # get directly from sumo
        reward, reward_detail_dict = self.get_rewards_from_sumo(tfl, tfl.dic_vehicles, action)
        return reward*(1-0.8), reward_detail_dict

    

    
