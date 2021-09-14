# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace
# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation

import os
import pygame
import time
import numpy as np
import nengo
import nengo_loihi
from nengo.neurons import Direct, LIF
from nengo.simulator import Simulator as NengoSimulator
from IK import *
from utilities import *

os.environ["KAPOHOBAY"] = "1"
nengo_loihi.set_defaults()

class PS4Controller():
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    def __init__(self, use_keyboard):
        """Initialize the joystick components"""
        if not use_keyboard:
            pygame.init()
            pygame.joystick.init()
            self.controller = pygame.joystick.Joystick(0)
            self.controller.init()

        self.axis_direction = None
        self.time = time.time()

        self.buttons = {
                    0: {'key':"Cross",    'value': False},
                    1: {'key':"Circle",   'value': False},
                    2: {'key':"Square",   'value': False},
                    3: {'key':"Triangle", 'value': False},
                    4: {'key':"Share",    'value': False},
                    5: {'key':"P",        'value': False},
                    6: {'key':"Options",  'value': False},
                    7: False,
                    8: False,
                    9:  {'key':"L1",      'value': False},
                    10: {'key':"R1",      'value': False},
                    11: {'key':"Up",      'value': False},
                    12: {'key':"Down",    'value': False},
                    13: {'key':"Left",    'value': False},
                    14: {'key':"Right",   'value': False},
                    15: {'key':"Keypad",  'value': False}}

        self.axis = {
                    0: {'key': 'Left_horizontal',  'value': 0.0},
                    1: {'key': 'Left_vertical',    'value': 0.0},
                    2: {'key': 'Right_horizontal', 'value': 0.0},
                    3: {'key': 'Right_vertical',   'value': 0.0},
                    4: {'key': 'L2', 'value': -1.0},
                    5: {'key': 'R2', 'value': -1.0}

        }


    def debug(self, robot_state, arm, actuation_function_axis):
        default = 0.3

        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:

                    x_0 , y_1 = self.controller.get_axis(0), self.controller.get_axis(1) # Left joistick
                    '''
                    # For Windows
                    x_2 , y_3 = self.controller.get_axis(2), self.controller.get_axis(3) # Right joistick
                    '''

                    # For Linux
                    x_2 , y_3 = self.controller.get_axis(3), self.controller.get_axis(4) # Right joistick

                    print("x_0: ", x_0)
                    print("y_1: ", y_1)
                    print("x_2: ", x_2)
                    print("y_3: ", y_3)
                    print()

                   

                      
    def generate_net(self,nengo_type,
            probes_on=True, # set True to record data
            n_scale=3000,
            ):

        net = nengo.Network(seed=0)
        config = nengo.Config(nengo.Connection, nengo.Ensemble)
        np.random.seed(0)
        with net, config:
            

            def target_xyz_func(t):
                return self.target

            def current_xyz_func(t):
                return self.current

            net.axis = 3
            axis = net.axis  # the number axis
            net.probes_on = probes_on

            net.xyz_target = nengo.Node(target_xyz_func,   label='xyz_in')
            net.xyz_current = nengo.Node(current_xyz_func, label='xyz_current')

            if nengo_type == "Direct":    
                net.diff = nengo.Ensemble(
                    n_neurons=n_scale, dimensions=axis,
                    radius=np.sqrt(axis),
                    neuron_type=Direct(),

                    )
            else:
                net.diff = nengo.Ensemble(
                    n_neurons=n_scale, dimensions=axis,
                    radius=np.sqrt(axis),
                    neuron_type=LIF(),
                    )

            nengo.Connection(net.xyz_target,  net.diff)     
            nengo.Connection(net.xyz_current, net.diff, transform=-1) 
            
            if probes_on:
                net.probe_diff   = nengo.Probe(net.diff, synapse=0.05)

            def output_func(t, x):
                self.output = np.copy(x)

            output = nengo.Node(output_func, size_in=axis, size_out=0)
            nengo.Connection(net.diff, output) 

        if nengo_type == "LIF LOIHI":
        # LOIHI
            self.sim = nengo_loihi.Simulator(net,remove_passthrough=False, target='loihi', hardware_options={
                    "snip_max_spikes_per_step": 300
                    })
        else:
        # NENGO    
            self.sim = nengo.Simulator(net, dt=0.001)

            
    def inverse_kinematics(self, nengo_type, lr = 1e-3,n_scale=500):
        model = nengo.Network(seed=0)
        if nengo_type == "Direct":
            neuron = Direct()
        else:
            neuron = LIF()
            
        np.random.seed(0)
        model.config[nengo.Ensemble].neuron_type = nengo.Direct()
        with model:

            def target_xyz_func(t):
                return self.target 

            def target_abg_func(t):
                return self.abg_target 

            def current_q_func(t):
                return self.current_q

            model.q_in = nengo.Node(current_q_func)
            model.q_c = nengo.Ensemble(n_scale*5,
                                dimensions=5,
                                radius = np.sqrt(5),
                                neuron_type=LIF(),)

            nengo.Connection(model.q_in, model.q_c,function=lambda x: x / (2*np.pi))

            model.q_t = nengo.Ensemble(n_scale*5, dimensions=5,
                                radius = np.sqrt(5), 
                                )
            model.conn = nengo.Connection(model.q_c, model.q_t, synapse=0.01)

            model.xyz_t = nengo.Ensemble(n_scale*3, dimensions=3,
                                radius = np.sqrt(3),
                                neuron_type=neuron,
                                )
            model.abg_t = nengo.Ensemble(n_scale*3, dimensions=3,
                              radius = np.sqrt(3),
                              )


            def q2xyz(q):
                q_ = q * (2*np.pi)
                t = calc_T(q_)
                return t[0], t[1], t[2]

            nengo.Connection(model.q_t, model.xyz_t, function=q2xyz, synapse=0.01)

            model.xyz_in = nengo.Node(target_xyz_func)
            nengo.Connection(model.xyz_in, model.xyz_t, transform=-1, synapse=0.01)

            model.abg_in = nengo.Node(target_abg_func)
            nengo.Connection(model.abg_in, model.abg_t, function=lambda x: x / (2*np.pi))

            model.error_node = nengo.Node(size_in=11)
            model.error_q = nengo.Ensemble(n_scale*11, dimensions=11,
                                    intercepts=get_intercepts(n_scale*11, 11),
                                    radius = np.sqrt(11), 
                                    )

            def combine(error_q):
                error_q_ = np.zeros(11)
                error_q_[0:5] = error_q[0:5] * (2*np.pi)   # current
                error_q_[5:8] = error_q[5:8]               # xyz
                error_q_[8:] = error_q[8:] * (2*np.pi)     # abg target

                direction = np.zeros(6)
                direction[:3] = error_q[5:8]
                R_e = calculate_R(error_q[0:5])
                direction[3:] = calc_orientation_forces(error_q_[8:], R_e)

                direction = direction[self.control_dof]
                J_x = calc_J_O(error_q_[0:5])
                J_x = J_x[self.control_dof]
                tmp = np.dot(np.linalg.pinv(J_x), direction)
                return tmp / (2*np.pi)

            nengo.Connection(model.q_t, model.error_node[0:5])
            nengo.Connection(model.xyz_t, model.error_node[5:8])
            nengo.Connection(model.abg_t, model.error_node[8:])

            nengo.Connection(model.error_node, model.error_q)

            model.error_combined = nengo.Ensemble(n_scale*5, dimensions=5,
                                            intercepts=get_intercepts(n_scale*5, 5,"error_combined"),
                                            radius = np.sqrt(5),
                                            neuron_type=neuron,
                                        )
            nengo.Connection(model.error_q, model.error_combined, function=combine, synapse=0.01)

            model.conn.learning_rule_type = nengo.PES(learning_rate=lr)
            nengo.Connection(model.error_combined, model.conn.learning_rule)
            
            

            def comp_error(error_combined):
                return np.sqrt(sum(np.power(error_combined, 2)))

            model.error_out = nengo.Node(size_in=1, size_out=1)
            nengo.Connection(model.error_combined, model.error_out, function=comp_error)

            def output_func(t, x):
                self.output_q = np.copy(x)

            output = nengo.Node(output_func, size_in=5, size_out=0)
            nengo.Connection(model.q_t, output) 

        if nengo_type == "LIF LOIHI":
        # LOIHI
            self.sim = nengo_loihi.Simulator(model,remove_passthrough=False, target='loihi', hardware_options={
                    "snip_max_spikes_per_step": 300
                    })
        else:
        # NENGO    
            self.sim = nengo.Simulator(model, dt=0.001, optimize=True)


    # Stop sending commends after leaving the joystick 
    def get_loihi_time(self):
        '''
        # 0.01 ms
        if self.axis_direction == "Right":
            return 0.39
        else:
            return 0.37
        '''

        # 0.2 ms / 0.1 ms
        if self.axis_direction == "Right":
            return 0.76
        if self.axis_direction == "Left":
            return 0.74
        elif self.axis_direction == "Backward":
            return 0.85
        '''

        # 0.05 ms
        if self.axis_direction == "Backward":
            return 0.85
        '''    
        return 0.73
        
        
        

    def listen_axis(self, robot_state, arm, nengo_type, use_keyboard, os_type, IK_model, actuation_function_axis):
        dict_nengo = {"no nengo": 0.144, "Direct": 0.352, "LIF OPENU":0.366, "LIF ALYN": 0.479, "LIF LOIHI":self.get_loihi_time()}
        self.last_time = time.time()
        self.this_time = None
        self.abg_target = None

        self.task_list = ['Home', 'Before_Destruct', 'Home', 'Drinking','Shelf','Floor','Shelf','Home','Before_Destruct']
    #    self.task_list = ['Home', 'Drinking','Home','Before_Destruct']

        self.task = 0 # Home, Floor, Shelf or Drinking
        self.control_dof = [True,True,True,False,False,False]
        self.reaching_to_target = None # True = reaching to target, False = reaching to chair, None = default position

        
        if nengo_type == "no nengo":
            while True:
                if not use_keyboard:
                    self.listen_joystick(actuation_function_axis, robot_state, arm, nengo_type, dict_nengo, os_type, IK_model)
                else:
                   self.listen_keyboard(actuation_function_axis, robot_state, arm, nengo_type, dict_nengo, os_type, IK_model) 
        else:
            if IK_model == 1:
                # prepare nengo model
                self.target = np.zeros(3)
                self.current = np.zeros(3)
                self.generate_net(nengo_type)
                with self.sim:
                    while True:
                        if not use_keyboard:
                            self.listen_joystick(actuation_function_axis, robot_state, arm, nengo_type, dict_nengo, os_type, IK_model)
                        else:
                            self.listen_keyboard(actuation_function_axis, robot_state, arm, nengo_type, dict_nengo, os_type, IK_model) 
            elif IK_model == 2:
                 # prepare nengo model
                self.target = np.zeros(3)
                self.current = np.zeros(3)
                self.abg_target = np.zeros(3)
                self.current_q = robot_state.state_model
                self.inverse_kinematics(nengo_type)
                with self.sim:
                    while True:
                        if not use_keyboard:
                            self.listen_joystick(actuation_function_axis, robot_state, arm, nengo_type, dict_nengo, os_type, IK_model)
                        else:
                            self.listen_keyboard(actuation_function_axis, robot_state, arm, nengo_type, dict_nengo, os_type, IK_model)
                

                

                    

    def listen_joystick(self, actuation_function_axis, robot_state, arm, nengo_type, dict_nengo, os_type, IK_model):
        joystick_sensitivity = 0.9 
        act = False

        if os_type == "xavier":
            self.joy = 0 # 1 = axis 0,1 (up,down) | 0 = axis 3,4 (left,right,forward,backword)
            notice = False

        for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    act = False
                    self.this_time = time.time()
                    x_0 , y_1 = self.controller.get_axis(0), self.controller.get_axis(1)
                    x_2 = x_0
                    y_3 = y_1

                    if os_type == "ubuntu_18":
                        x_2 , y_3 = self.controller.get_axis(3), self.controller.get_axis(4) # Right joistick
                        '''
                        # For Windows
                        x_2 , y_3 = self.controller.get_axis(2), self.controller.get_axis(3) # Right joistick
                        '''

                    if os_type == "ubuntu_18" or (os_type == "xavier" and self.joy == 1):
                        if x_0 > joystick_sensitivity or x_0 < -joystick_sensitivity: # Orientation Left or Right
                            act = True
                            if x_0 < joystick_sensitivity: # Orientation Left
                                self.axis_direction = "o_left"
                            else: # Orientation Right
                                self.axis_direction = "o_right"

                        

                        elif y_1 > joystick_sensitivity or y_1 < -joystick_sensitivity: # Up or Down
                            act = True
                            if y_1 < joystick_sensitivity: # Up
                                self.axis_direction = "Up"
                            else: # Down
                                self.axis_direction = "Down"



                    if os_type == "ubuntu_18" or (os_type == "xavier" and self.joy == 0):
                        if x_2 > joystick_sensitivity or x_2 < -joystick_sensitivity: # Left or Right
                            act = True
                            if x_2 < joystick_sensitivity: # Left
                                self.axis_direction = "Left"
                            else: # Right
                                self.axis_direction = "Right"

                            

                        elif y_3 > joystick_sensitivity or y_3 < -joystick_sensitivity: # Forward or Backward
                            act = True
                            if y_3 < joystick_sensitivity: # Forward
                                self.axis_direction = "Forward"
                            else: # Backward
                                self.axis_direction = "Backward"        

                    

                elif event.type == pygame.JOYBUTTONDOWN:
                    try:
                        self.buttons[event.button]['value'] = True
                        self.axis_direction = None

                        if os_type == "xavier":
                            if event.button ==4 or event.button==5: notice=True # pay attention to buttons 4 and 5

                    except:
                        print("Illegal button ", event.button)
                        continue
                elif event.type == pygame.JOYBUTTONUP:
                    try:
                        self.buttons[event.button]['value'] = False
                        self.axis_direction = None
                    except:
                        print("Illegal button ", event.button)
                        continue

                if actuation_function_axis is not None and \
                    self.this_time is not None:
                    time_tmp = self.this_time - self.last_time  # time between JOYAXISMOTION events
                    if time_tmp > dict_nengo[nengo_type] or (os_type == "xavier" and notice): 
                        if self.task_list[self.task] == 'Drinking':
                            self.abg_target = [1.5707963267948966, -0.08726646259971632, -1.570796326794897]
                            self.control_dof = [True,True,True,True,True,True]
                        elif self.task_list[self.task] == 'Shelf':
                            self.abg_target =  [0, 0, 0]
                            self.control_dof = [True,True,True,True,True,False]
                        else:
                            # default
                            self.abg_target = [3.141592653589793, 2.970679785877549e-16, -3.0543261909900767]
                            self.control_dof = [True,True,True,False,False,False]

                        actuation_function_axis(self, robot_state, act, self.axis_direction ,self.buttons, arm, time_tmp, os_type, nengo_type, IK_model)
                        self.last_time = self.this_time


    def listen_keyboard(self, actuation_function_axis,robot_state, arm, nengo_type, dict_nengo, os_type, IK_model):
        value = "start"
        act = False
        while value != 'q':
            value = input("To go right press: d \nTo go left press: a \nTo go forward press: w \nTo go backward prees: s \nTo go up press: u \nTo go down press: j \nTo shift task press: t \nTo shift routine press: r \nTo terminate press: q \nTo close gripper: x \nTo open gripper: o\n\n")
            if value == 'd':
                self.axis_direction = "Right"
            elif value == 'a':
                self.axis_direction = "Left"
            elif value == 'w':
                self.axis_direction = "Forward"
            elif value == 's':
                self.axis_direction = "Backward"
            elif value == 'u':
                self.axis_direction = "Up"
            elif value == 'j':
                self.axis_direction = "Down"    
            elif value == 't':
                self.buttons[3]['value'] = True
                self.axis_direction = None
            elif value == 'r':
                self.buttons[2]['value'] = True
                self.axis_direction = None
            elif value == 'x':
                self.buttons[0]['value'] = True
                self.axis_direction = None
            elif value == 'o':
                self.buttons[1]['value'] = True
                self.axis_direction = None
            else:
                print("Illegal value ",value) 
                continue
            
            act = True
            if actuation_function_axis is not None:                    
                    if self.task_list[self.task] == 'Drinking':
                        self.abg_target = [1.5707963267948966, -0.08726646259971632, -1.570796326794897]
                        self.control_dof = [True,True,True,True,True,True]
                    elif self.task_list[self.task] == 'Shelf':
                        self.abg_target =  [0, 0, 0]
                        self.control_dof = [True,True,True,True,True,False]
                    else:
                        # default
                        self.abg_target = [3.141592653589793, 2.970679785877549e-16, -3.0543261909900767]
                        self.control_dof = [True,True,True,False,False,False]

                    actuation_function_axis(self, robot_state, act, self.axis_direction ,self.buttons, arm, None, os_type, nengo_type, IK_model)
                    self.last_time = self.this_time