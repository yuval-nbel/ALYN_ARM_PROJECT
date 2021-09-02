from threading import current_thread
import importlib
import nengo
from nengo.neurons import Direct, LIF
from nengo.simulator import Simulator as NengoSimulator

from PS4_controller import PS4Controller

from RoboticArm import *

from IK import viper300
import numpy as np
import pprint
from utilities import *

import os
import nengo_loihi
os.environ["KAPOHOBAY"] = "1"
nengo_loihi.set_defaults()
import time

class RobotState:
    
    def __init__(self, init_state, openu):
        self.state_chair = init_state
        self.state_model = robot_to_model_position(init_state, openu)
    
    def update_chair(self, new_state, openu):
        self.state_chair = new_state
        self.state_model = robot_to_model_position(self.state_chair, openu)

    def update_model(self, update, openu):
        self.state_model += update
        m = model_to_robot_position(self.state_model, openu)
        for i in range(1,7):
            self.state_chair[i] = m[i]


"""""""""""""""""""""""""""""""""""""""""""""""""""""
Choose paramerters:
"""""""""""""""""""""""""""""""""""""""""""""""""""""

os_type = "ubuntu_18"           ## "ubuntu_18" / "xavier"  
openu = False                    ## True=openu, False=alyn
using_the_physical_arm = False
nengo_type = "LIF ALYN"        ## "no nengo" / "Direct" / "LIF OPENU" / "LIF ALYN" / "LIF LOIHI"
use_keyboard = True            ## True=keyboard, False=joystick
speed = 2                       ## effects the speed 
print_diff = True               ## True=print the difference calculation / False = don't print
IK_model = 2                    ## 1=Hybrid, 2=SNN

"""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""

robot_config = return_Robot(openu, speed)    # Viper300 configuration
velocity_delta = 0.01  # Gain factor for actuation
ik_model = viper300()   # Viper300 IK model
state = RobotState(robot_config['Real']['Home'], openu)



if using_the_physical_arm:
    arm = RoboticArm(robot_config, COM_ID = '/dev/ttyUSB0')
    arm.go_home()
    # Dealing with the bug of the first move
    arm_actuation = robot_config['Real']['Home']
    state.update_chair(arm_actuation)   
    arm.set_position(arm_actuation)


#####################
# Note:
# state.state_chair = engines_position
# state.state_model = joint_position
#####################

print('Home position: {}, at: {}'.format(state.state_chair, state.state_model))

def get_xyz_numeric_3d(axis):
    return np.array([axis[0][0],axis[1][0],axis[2][0]])

# So the arm won't hurt the person in the chair
def in_limit(target):
    if target[1] <= -0.15: # Y axis limit
        if target[0] >= 0.24: # X axis limit
            print("You have reached a limit!")
            return False
    
    return True



position = state.state_model
reference = get_xyz_numeric_3d(ik_model.get_xyz_numeric(position))
last_state = None
new_state = None
change = 0
print("current xyz: ", reference)
print("##################################################################")
print("#################### reference determined ########################")
print("##################################################################")



    

def actuation_function_axis(self, robot_state, act, axis_direction, buttons_dict, arm, time_tmp, os_type, nengo_type,IK_model=1) :

    global reference
    global last_state
    global new_state
    global robot_config

    position = state.state_model

    self.current = get_xyz_numeric_3d(ik_model.get_xyz_numeric(position))
    self.current_q = position
        
    # Right
    if axis_direction == "Right": # Right stick -> right
        new_state = "Right"

        if last_state != new_state and last_state is not None:
            reference = self.current

        self.target = [0.5, reference[1] ,reference[2]]

        last_state = new_state




    # Left
    elif axis_direction == "Left": # Right stick -> left
        new_state = "Left"

        if last_state != new_state and last_state is not None:
            reference = self.current

        self.target = [-0.5, reference[1] ,reference[2]]

        last_state = new_state
        
            
        

       
    # Forward 
    elif axis_direction == "Forward": # Right stick -> up
        new_state = "Forward"
        
        if last_state != new_state and last_state is not None:
            reference = self.current

        self.target = [reference[0], 0.7, reference[2]]

        last_state = new_state
        
        


    # Backward
    elif axis_direction == "Backward": # Right stick -> down
        new_state = "Backward"
        
        if last_state != new_state and last_state is not None:
            reference = self.current

        self.target = [reference[0], -0.7, reference[2]]

        last_state = new_state
        

    # Up 
    elif axis_direction == "Up": # Left stick -> up
        new_state = "Up"
        
        if last_state != new_state and last_state is not None:
            reference = self.current

        self.target = [reference[0], reference[1] ,0.9]

        last_state = new_state


    # Down
    elif axis_direction == "Down": # Left stick -> down
        new_state = "Down"
        
        if last_state != new_state and last_state is not None:
            reference = self.current

        self.target = [reference[0], reference[1] ,-0.6]

        last_state = new_state


    # gripper - Working with self.current
    elif ((os_type == "ubuntu_18" or use_keyboard) and buttons_dict[1]['value']) or \
          (os_type == "xavier" and buttons_dict[2]['value']):   # Circle press
        # Set Goal Tourqe to 20
        arm.change_current_gripper(20)
        print("open gripper")
        
    
    elif ((os_type == "ubuntu_18" or use_keyboard) and buttons_dict[0]['value']) or \
          (os_type == "xavier" and buttons_dict[1]['value']):   # Cross press 
        # Set Goal Tourqe to -50
        arm.change_current_gripper(-50)
        print("close gripper")

    # Shift task
    elif ((os_type == "ubuntu_18" or use_keyboard) and buttons_dict[3]['value']) or \
          (os_type == "xavier" and buttons_dict[0]['value']):   # Rectangle press 

        # Checks if holding a cup
        arm_actuation = robot_state.state_chair
        grip = arm_actuation[9]
        '''
        if self.task_list[self.task] == 'Floor':   # So the arm won't hurt the chair on her way up from the floor
            if self.task_position == 'Default':
                self.reaching_to_target = True
                arm_actuation = robot_config['Real'][self.task_list[self.task]]['Target']
                arm_actuation[9] = grip
                self.task_position = 'Target'
                robot_state.update_chair(arm_actuation)
                if arm is not None:  
                    arm.set_position(arm_actuation)
                time.sleep(2)

            if self.task_position == 'Target':
                self.reaching_to_target = False
                arm_actuation = robot_config['Real'][self.task_list[self.task]]['Chair']
                arm_actuation[9] = grip
                self.task_position = 'Chair'
                robot_state.update_chair(arm_actuation)
                if arm is not None:  
                    arm.set_position(arm_actuation)
                time.sleep(1)
        '''

        self.task = (self.task + 1) % len(self.task_list)
        self.task_position = 'Default' 

        try:
            self.reaching_to_target = None
            arm_actuation = robot_config['Real'][self.task_list[self.task]][self.task_position]
        except:
            arm_actuation = robot_config['Real'][self.task_list[self.task]]

        arm_actuation[9] = grip
        robot_state.update_chair(arm_actuation)  
        
        if arm is not None:  
            arm.set_position(arm_actuation)

        last_state = "Shift task"

        print("Shift task to: ", self.task_list[self.task])
        pprint.pprint('new engines position: {}'.format(arm_actuation))
        pprint.pprint('chair state: {}'.format(robot_state.state_chair))
        pprint.pprint('model state: {}'.format(get_xyz_numeric_3d(ik_model.get_xyz_numeric(robot_state.state_model))))

   # task routine
    elif ((os_type == "ubuntu_18" or use_keyboard) and buttons_dict[2]['value']) or \
          (os_type == "xavier" and buttons_dict[3]['value']):   # Triangle press  
        arm_actuation = robot_state.state_chair
        grip = arm_actuation[9]
        try:
            

            if self.reaching_to_target is None:         
                self.reaching_to_target = True
                arm_actuation = robot_config['Real'][self.task_list[self.task]]['Target']
                arm_actuation[9] = grip
                self.task_position = 'Target'
            elif self.reaching_to_target == True:
                self.reaching_to_target = False
                arm_actuation = robot_config['Real'][self.task_list[self.task]]['Chair']
                arm_actuation[9] = grip
                self.task_position = 'Chair'
            elif self.reaching_to_target == False:
                self.reaching_to_target = None
                arm_actuation = robot_config['Real'][self.task_list[self.task]]['Default']
                arm_actuation[9] = grip
                self.task_position = 'Default'

            robot_state.update_chair(arm_actuation)

            if arm is not None:  
                arm.set_position(arm_actuation)

            last_state = "Shift routine"


            print("Reaching to: ", self.task_position)
            pprint.pprint('new engines position: {}'.format(arm_actuation))
            pprint.pprint('chair state: {}'.format(robot_state.state_chair))
            pprint.pprint('model state: {}'.format(get_xyz_numeric_3d(ik_model.get_xyz_numeric(robot_state.state_model))))
        except:
            print("TASK HAS ONLY DEFAULT POSITION: ", self.task_list[self.task])
 


    if os_type == "xavier":
        # change joystick
        if buttons_dict[4]['value']: # L1 
            if self.joy == 0: self.joy = 1
            else: self.joy = 0
            print("joystick changed")

        # destruct
        elif buttons_dict[5]['value']: # Down press -> Terminate
            arm.destruct()
            return

  

    if axis_direction != "o_left" and axis_direction != "o_right" and axis_direction is not None and act:

        
        # NENGO
        if nengo_type != "no nengo" and nengo_type != "LIF LOIHI":
            
            TS = 1
            start = time.time()
            self.sim.run(TS, progress_bar=True)
            end = time.time()
            print("#########################") # inference time sim
            print('inference time sim:')
            print(end-start)
            print("#########################")
            

        # LOIHI
        if nengo_type == "LIF LOIHI":
            TS = 0.1
            start = time.time()
            self.sim.run(TS)
            end = time.time()
            print("#########################") # inference time loihi
            print('inference time loihi:')
            print(end-start)
            print("#########################")
        

        print("#########################") # time between JOYAXISMOTION events
        print('time between JOYAXISMOTION events:')
        print(time_tmp)
        print("#########################")

        # Use the hybrid model
        if IK_model == 1 or nengo_type == "no nengo":

            target = self.target + self.abg_target # include oreintation in target

            direction = np.zeros(6)

            if nengo_type != "no nengo":
                # LOIHI / NENGO
                direction[:3] = self.output
            else:
                # Numeric
                direction[:3] = target[:3] - self.current 
            
            if np.sum(self.control_dof[3:]) > 0:
                R_e = ik_model.calculate_R(position)
                direction[3:] = calc_orientation_forces(target[3:], R_e)

            direction = direction[self.control_dof]

            J = ik_model.calc_J_numeric(position)
            J = J[self.control_dof]
        
            updated_position = (np.dot(np.linalg.pinv(J), direction)*velocity_delta)

            if print_diff:
                ###########
                # Format the old state to make a difference calculation
                state_before = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7:0, 8:0, 9:0}
                for i in range(1,9):
                    state_before[i] = robot_state.state_chair[i]
                ###########

        else:
            updated_position = self.output_q - self.current_q

        robot_state.update_model(updated_position, openu)
        arm_actuation = robot_state.state_chair

        updated_current = get_xyz_numeric_3d(ik_model.get_xyz_numeric(robot_state.state_model))
        
        if not in_limit(updated_current):
            robot_state.update_model(-updated_position, openu)
            arm_actuation = robot_state.state_chair

        if print_diff:
            ###########
            # Format the new state to make a difference calculation
            print()
            state_after = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7:0, 8:0, 9:0}
            for i in range(1,9):
                state_after[i] = robot_state.state_chair[i]
            print("############# diff #############")
            tmp = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7:0, 8:0, 9:0}
            for i in range(1,9):
                tmp[i] = state_after[i] - state_before[i]

            # Print difference calculation        
            print(tmp)
            print("################################")
            print()
            ###########

        pprint.pprint('Target: {}'.format(target))
        pprint.pprint('new engines position: {}'.format(arm_actuation))
        pprint.pprint('at position: {}'.format(updated_current))
        pprint.pprint('at state: {}'.format(robot_state.state_model))

        print(axis_direction)

        if arm is not None:
            arm.set_position(arm_actuation)

        self.axis_direction = None
    
        buttons_dict[0]['value'] = False
        buttons_dict[1]['value'] = False
        buttons_dict[2]['value'] = False
        buttons_dict[3]['value'] = False
        
        if os_type == "xavier":
            buttons_dict[4]['value'] = False

        axis_direction = None

    
    # orientation of gripper
    elif axis_direction == "o_left" or axis_direction == "o_right" and act:
        arm_actuation = robot_state.state_chair
        if axis_direction == "o_left":
            arm_actuation[7] += 1 
            print("Orientation left")
        else:
            arm_actuation[7] -= 1 
            print("Orientation right")
        pprint.pprint('new engines position: {}'.format(arm_actuation))
        
        robot_state.update_chair(arm_actuation)  

        if arm is not None:  
            arm.set_position(arm_actuation)

    

    self.axis_direction = None
    
    buttons_dict[0]['value'] = False
    buttons_dict[1]['value'] = False
    buttons_dict[2]['value'] = False
    buttons_dict[3]['value'] = False

    if os_type == "xavier":
            buttons_dict[4]['value'] = False

    axis_direction = None


PS4 = PS4Controller(use_keyboard)

if not using_the_physical_arm:
    PS4.listen_axis(state, None, nengo_type, use_keyboard, os_type, actuation_function_axis = actuation_function_axis)
if using_the_physical_arm:    
    PS4.listen_axis(state, arm, nengo_type, use_keyboard, os_type, actuation_function_axis = actuation_function_axis)

#PS4.debug(state, None, actuation_function_axis = actuation_function_axis)