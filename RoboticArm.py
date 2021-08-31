# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace
# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
# pylint: disable=invalid-name

"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphoc Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 7.9.2020

This work is using Robotis' Dynamixel SDK package (pip install dynamixel-sdk) 
and it is utilized here to control Trossen Robotics' arms. 

Code was tested on the: 
1. 5DOF WidowX 200 (https://www.trossenrobotics.com/widowx-200-robot-arm.aspx)
2. 6DOF ViperX 300 (https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx)
"""

import time
import numpy as np
from dynamixel_sdk import *

# Offset angles for the physical arm in relative to the IK mpdel
  
offset_relative_to_IK_Model = {1: 90, 2: 180, 3: 180, 4: 180, 
                               5: 180, 6: 0, 7: 180, 8: 0, 9: 0}

offset_relative_to_IK_Model_openu = {1: 181, 2: 180, 3: 180, 4: 180, 
                               5: 180, 6: 0, 7: 180, 8: 0, 9: 0}

def robot_to_model_position(robot_position, openu=False):
    if openu:
        offset = offset_relative_to_IK_Model_openu
    else:
        offset = offset_relative_to_IK_Model
    return [     np.deg2rad(robot_position[1]-offset[1]),
            -1 * np.deg2rad(robot_position[3]-offset[3]),
            -1 * np.deg2rad((360-robot_position[4])-offset[4]),
                 np.deg2rad(robot_position[6]-offset[6]),
            -1 * np.deg2rad(robot_position[7]-offset[7])]

def model_to_robot_position(model_position, openu=False):
    if openu:
        offset = offset_relative_to_IK_Model_openu
    else:
        offset = offset_relative_to_IK_Model
    f = [ (np.rad2deg(     model_position[0])+offset[1])%360,
          (np.rad2deg(-1 * model_position[1])+offset[2])%360,
          360-((np.rad2deg(-1 * model_position[2])+offset[4])%360),
          (np.rad2deg(     model_position[3])+offset[6])%360,
          (np.rad2deg(-1 * model_position[4])+offset[7])%360]

    return {1: f[0], 2: 361-f[1], 3: f[1], 4: f[2],
            5: 361-f[2], 6: f[3], 7: f[4], 8: 180, 9: 180}

# ALYN    
full_ang1 = 361
full_ang2 = 361

# OPENU
full_ang1_openu = 360
full_ang2_openu = 362


Robot = {'Real':{
            'CMD':
                {'Baud Rate'        : {'Address': 8, 'Value': {9600   : 0,
                                                               57600  : 1,
                                                               115200 : 2,
                                                               1000000: 3,
                                                               2000000: 4,
                                                               3000000: 5,
                                                               4000000: 6,
                                                               4500000: 7}},                                               
                 'Operating mode'   : {'Address': 11, 'Value': {'Torque'   : 0,
                                                                'Velocity' : 1,
                                                                'Position' : 3,
                                                                'PWM'      : 16}},
                 'Secondary(Shadow) ID'   : {'Address': 12, 'Value': {'Disable'   : 255}},                                               
                 'Torque Enable'    : {'Address': 64, 'Value': {'OFF': 0, 'ON' : 1}},                           
                 'LED'              : {'Address': 65, 'Value': {'OFF': 0, 'ON' : 1}},                                  
                 'Goal Position'    : {'Address': 116},                         
                 'Present Position' : {'Address': 132}, 
                 'Goal torque'      : {'Address': 102},
                 'Ranges'           : {1: range (0,   360),
                                       2: range (75,  290),
                                       3: range (75,  290),
                                       4: range (85,  285),
                                       5: range (85,  285),
                                       6: range (0,   360),
                                       7: range (55,  275),
                                       8: range (20,  320),
                                       9: range (130, 260)},
                 'Goal PWM'   : {'Address': 100, 'Value': 885}, # ranging [-885, 885] # ACTUALLY CONTROL THE SPEED
                 'Goal Current'   : {'Address': 102, 'Value': 20}, # Maximum is the Limit torque
                 'Velocity Limit'   : {'Address': 44, 'Value': 100}, # ranging [0, 1023]
                 'Limit torque'     : {'Address': 38,  'Value': 50},  # ranging [-1193, 1193], 2.69mA per step, 3.210A
                 'Profile Velocity'      : {'Address': 112, 'Value': 1000}, 
                 'Profile Acceleration'  : {'Address': 108, 'Value': 2}, # speed (not high from ProfileVelocity/2)
                'Max Voltage Limit'     : {'Address': 32,  'Value': 120}, # 9.0 ~ 12.0 [V] (Recommended : 11.1V)
                'Min Voltage Limit'     : {'Address': 34,  'Value': 90},
                },
                'Priority': [[4, 5], [2, 3], [1], [6], [8], [9], [7]],
             # Note that engines 2 and 5 were set to reverse mode to allow 
             # both to be configured similarly to their counterpart.
             'Home'    : {1: 85, 2: full_ang1-135, 3: 135, 4: 180, 5: full_ang2-180, 6: 180, 7:135, 8:180, 9:255},
             'Drinking'    :   {'Default': {1: 85, 2: full_ang1-170, 3: 170, 4: 170, 5: full_ang2-170, 6: 270, 7:256, 8:90, 9:255},
                                'Target' : {1: 88, 2: 132, 3: 229, 4: 229, 5: 132, 6: 265, 7: 255, 8: 90, 9: 255}, # DESK                                                           
                                'Chair'  : {1: 24, 2: 192, 3: 169, 4: 182, 5: 179, 6: 255, 7: 256, 8: 90, 9: 255}},

            
            'Shelf'       :   {'Default': {1: 85, 2: full_ang1-145, 3: 145, 4: 190, 5: full_ang2-190, 6: 180, 7:135, 8:180, 9:255},
                                'Target' : {1: 111.31702715038168, 2: 177.14316797592414, 3: 183.85683202407586, 4: 191.04324289760845, 5: 169.95675710239155, 6: 179.45891861047554, 7: 170, 8: 180, 9: 255},  
                                'Chair'  : {1: 38, 2: 250, 3: full_ang1-250, 4: 117, 5: full_ang2-117, 6: 273, 7: 204, 8: 89, 9: 255},
                        },

            'Floor'       :   {'Default': {1: 175, 2: 193, 3: 168, 4: 191, 5: 170, 6: 178, 7: 156, 8: 187, 9: 255},
                                'Target' : {1: 182.7964620728535, 2: 101.20365743496313, 3: 259.79634256503687, 4: 188.74850042860834, 5: 172.25149957139166, 6: 179.92160867784926, 7: 156, 8: 187, 9: 255},
                                'Chair'  : {1: 38, 2: 250, 3: full_ang1-250, 4: 117, 5: full_ang2-117, 6: 273, 7: 204, 8: 89, 9: 255}},

             'Before_Destruct' : {1: 84, 2: full_ang1-75, 3: 75,  4: 90, 5: full_ang2-90, 6: 180, 7:135, 8:180, 9:255},
                      }
        } 



Robot_openu = {'Real':{
            'CMD':
                {'Baud Rate'        : {'Address': 8, 'Value': {9600   : 0,
                                                               57600  : 1,
                                                               115200 : 2,
                                                               1000000: 3,
                                                               2000000: 4,
                                                               3000000: 5,
                                                               4000000: 6,
                                                               4500000: 7}}, 
                 'Operating mode'   : {'Address': 11, 'Value': {'Torque'   : 0,
                                                                'Velocity' : 1,
                                                                'Position' : 3,
                                                                'PWM'      : 16}},
                 'Torque Enable'    : {'Address': 64, 'Value': {'OFF': 0, 'ON' : 1}},                           
                 'LED'              : {'Address': 65, 'Value': {'OFF': 0, 'ON' : 1}},                                  
                 'Goal Position'    : {'Address': 116},                         
                 'Present Position' : {'Address': 132}, 
                 'Goal torque'      : {'Address': 102},
                 'Ranges'           : {1: range (0,   360),
                                       2: range (75,  290),
                                       3: range (75,  290),
                                       4: range (85,  285),
                                       5: range (85,  285),
                                       6: range (0,   360),
                                       7: range (55,  275),
                                       8: range (20,  320),
                                       9: range (130, 260)},
                 'Goal PWM'   : {'Address': 100, 'Value': 885}, # ranging [-885, 885]
                 'Goal Current'   : {'Address': 102, 'Value': 0}, # Maximum is the Limit torque
                 'Limit torque'     : {'Address': 38,  'Value': 100},  # ranging [-1193, 1193], 2.69mA per step, 3.210A
                 'Velocity Limit'     : {'Address': 44,  'Value': 100},
                 'Profile Velocity'      : {'Address': 112, 'Value': 30000}, 
                 'Profile Acceleration'  : {'Address': 108, 'Value': 1}, # speed (not high from ProfileVelocity/2)
                },
             'Priority': [[4, 5], [2, 3], [1], [6], [7], [8], [9]],
             
             # Note that engines 2 and 5 were set to reverse mode to allow 
             # both to be configured similarly to their counterpart.
             'Home'        :   {1: 180, 2: full_ang1_openu-135, 3: 135, 4: 180, 5: full_ang2_openu-180, 6: 180, 7:135, 8:180, 9:240},

             'Drinking'    :   {'Default': {1: 180, 2: full_ang1_openu-170, 3: 170, 4: 170, 5: full_ang2_openu-170, 6: 90, 7:90, 8:90, 9:240},
                                'Target' : {1: 180, 2: full_ang1_openu-248, 3: 248, 4: 250, 5: full_ang2_openu-250, 6: 88, 7: 90, 8: 90, 9: 240},
                                'Chair'  : {1: 123, 2: full_ang1_openu-154, 3: 154, 4: 155, 5: full_ang2_openu-155, 6: 91, 7: 90, 8: 90, 9: 240}},

             'Shelf'       :   {'Default': {1: 180, 2: full_ang1_openu-145, 3: 145, 4: 190, 5: full_ang2_openu-190, 6: 180, 7:135, 8:180, 9:255},
                                'Target' : {1: 178, 2: 193, 3: 168, 4: 192, 5: 170, 6: 179, 7: 172, 8: 180, 9: 255},
                                'Chair'  : {1: 133, 2: 250, 3: 111, 4: 117, 5: 244, 6: 273, 7: 204, 8: 89, 9: 255}},
            'Before_Destruct' : {1: 180, 2: full_ang1_openu-75, 3: 75,  4: 90, 5: full_ang2_openu-90, 6: 180, 7:135, 8:180, 9:240},
     
           
             'Floor'       :   {'Default': {1: 180, 2: 193, 3: 168, 4: 191, 5: 170, 6: 178, 7: 156, 8: 187, 9: 255},
                                'Target' : {1: 180, 2: 105.45228363449328, 3: 255.54771636550672, 4: 186.10935569252223, 5: 174.89064430747777, 6: 180.0302012302459, 7: 156, 8: 187, 9: 255},
                                'Chair'  : {1: 133, 2: 250, 3: full_ang1_openu-250, 4: 117, 5: full_ang2_openu-117, 6: 273, 7: 204, 8: 89, 9: 255}},
                              
             
            }
        } 

def return_Robot(openu, speed):
    if openu:  
        Robot = {'Real':{
            'CMD':
                {'Baud Rate'        : {'Address': 8, 'Value': {9600   : 0,
                                                               57600  : 1,
                                                               115200 : 2,
                                                               1000000: 3,
                                                               2000000: 4,
                                                               3000000: 5,
                                                               4000000: 6,
                                                               4500000: 7}}, 
                 'Operating mode'   : {'Address': 11, 'Value': {'Torque'   : 0,
                                                                'Velocity' : 1,
                                                                'Position' : 3,
                                                                'PWM'      : 16}},
                 'Torque Enable'    : {'Address': 64, 'Value': {'OFF': 0, 'ON' : 1}},                           
                 'LED'              : {'Address': 65, 'Value': {'OFF': 0, 'ON' : 1}},                                  
                 'Goal Position'    : {'Address': 116},                         
                 'Present Position' : {'Address': 132}, 
                 'Goal torque'      : {'Address': 102},
                 'Ranges'           : {1: range (0,   360),
                                       2: range (75,  290),
                                       3: range (75,  290),
                                       4: range (85,  285),
                                       5: range (85,  285),
                                       6: range (0,   360),
                                       7: range (55,  275),
                                       8: range (20,  320),
                                       9: range (130, 260)},
                 'Goal PWM'   : {'Address': 100, 'Value': 885}, # ranging [-885, 885]
                 'Goal Current'   : {'Address': 102, 'Value': 0}, # Maximum is the Limit torque
                 'Limit torque'     : {'Address': 38,  'Value': 100},  # ranging [-1193, 1193], 2.69mA per step, 3.210A
                 'Velocity Limit'     : {'Address': 44,  'Value': 100},
                 'Profile Velocity'      : {'Address': 112, 'Value': 30000}, 
                 'Profile Acceleration'  : {'Address': 108, 'Value': speed}, # speed (not high from ProfileVelocity/2)
                },
             'Priority': [[4, 5], [2, 3], [1], [6], [7], [8], [9]],
             
             # Note that engines 2 and 5 were set to reverse mode to allow 
             # both to be configured similarly to their counterpart.
             'Home'        :   {1: 180, 2: full_ang1_openu-135, 3: 135, 4: 180, 5: full_ang2_openu-180, 6: 180, 7:135, 8:180, 9:240},

             'Drinking'    :   {'Default': {1: 180, 2: full_ang1_openu-170, 3: 170, 4: 170, 5: full_ang2_openu-170, 6: 90, 7:90, 8:90, 9:240},
                                'Target' : {1: 180, 2: full_ang1_openu-248, 3: 248, 4: 250, 5: full_ang2_openu-250, 6: 88, 7: 90, 8: 90, 9: 240},
                                'Chair'  : {1: 123, 2: full_ang1_openu-154, 3: 154, 4: 155, 5: full_ang2_openu-155, 6: 91, 7: 90, 8: 90, 9: 240}},

             'Shelf'       :   {'Default': {1: 180, 2: full_ang1_openu-145, 3: 145, 4: 190, 5: full_ang2_openu-190, 6: 180, 7:135, 8:180, 9:255},
                                'Target' : {1: 178, 2: 193, 3: 168, 4: 192, 5: 170, 6: 179, 7: 172, 8: 180, 9: 255},
                                'Chair'  : {1: 133, 2: 250, 3: 111, 4: 117, 5: 244, 6: 273, 7: 204, 8: 89, 9: 255}},
            'Before_Destruct' : {1: 180, 2: full_ang1_openu-75, 3: 75,  4: 90, 5: full_ang2_openu-90, 6: 180, 7:135, 8:180, 9:240},
     
           
             'Floor'       :   {'Default': {1: 180, 2: 193, 3: 168, 4: 191, 5: 170, 6: 178, 7: 156, 8: 187, 9: 255},
                                'Target' : {1: 180, 2: 105.45228363449328, 3: 255.54771636550672, 4: 186.10935569252223, 5: 174.89064430747777, 6: 180.0302012302459, 7: 156, 8: 187, 9: 255},
                                'Chair'  : {1: 133, 2: 250, 3: full_ang1_openu-250, 4: 117, 5: full_ang2_openu-117, 6: 273, 7: 204, 8: 89, 9: 255}},
                              
             
            }
        } 
    else:

        Robot = {'Real':{
            'CMD':
                {'Baud Rate'        : {'Address': 8, 'Value': {9600   : 0,
                                                               57600  : 1,
                                                               115200 : 2,
                                                               1000000: 3,
                                                               2000000: 4,
                                                               3000000: 5,
                                                               4000000: 6,
                                                               4500000: 7}},                                               
                 'Operating mode'   : {'Address': 11, 'Value': {'Torque'   : 0,
                                                                'Velocity' : 1,
                                                                'Position' : 3,
                                                                'PWM'      : 16}},
                 'Secondary(Shadow) ID'   : {'Address': 12, 'Value': {'Disable'   : 255}},                                               
                 'Torque Enable'    : {'Address': 64, 'Value': {'OFF': 0, 'ON' : 1}},                           
                 'LED'              : {'Address': 65, 'Value': {'OFF': 0, 'ON' : 1}},                                  
                 'Goal Position'    : {'Address': 116},                         
                 'Present Position' : {'Address': 132}, 
                 'Goal torque'      : {'Address': 102},
                 'Ranges'           : {1: range (0,   360),
                                       2: range (75,  290),
                                       3: range (75,  290),
                                       4: range (85,  285),
                                       5: range (85,  285),
                                       6: range (0,   360),
                                       7: range (55,  275),
                                       8: range (20,  320),
                                       9: range (130, 260)},
                 'Goal PWM'   : {'Address': 100, 'Value': 885}, # ranging [-885, 885] # ACTUALLY CONTROL THE SPEED
                 'Goal Current'   : {'Address': 102, 'Value': 20}, # Maximum is the Limit torque
                 'Velocity Limit'   : {'Address': 44, 'Value': 100}, # ranging [0, 1023]
                 'Limit torque'     : {'Address': 38,  'Value': 50},  # ranging [-1193, 1193], 2.69mA per step, 3.210A
                 'Profile Velocity'      : {'Address': 112, 'Value': 30000}, 
                 'Profile Acceleration'  : {'Address': 108, 'Value': speed}, # speed (not high from ProfileVelocity/2)
                'Max Voltage Limit'     : {'Address': 32,  'Value': 120}, # 9.0 ~ 12.0 [V] (Recommended : 11.1V)
                'Min Voltage Limit'     : {'Address': 34,  'Value': 90},
                },
                'Priority': [[4, 5], [2, 3], [1], [6], [8], [9], [7]],
             # Note that engines 2 and 5 were set to reverse mode to allow 
             # both to be configured similarly to their counterpart.
             'Home'    : {1: 85, 2: full_ang1-135, 3: 135, 4: 180, 5: full_ang2-180, 6: 180, 7:135, 8:180, 9:255},
             'Drinking'    :   {'Default': {1: 85, 2: full_ang1-170, 3: 170, 4: 170, 5: full_ang2-170, 6: 270, 7:256, 8:90, 9:255},
                                'Target' : {1: 88, 2: 132, 3: 229, 4: 229, 5: 132, 6: 265, 7: 255, 8: 90, 9: 255}, # DESK                                                           
                                'Chair'  : {1: 24, 2: 192, 3: 169, 4: 182, 5: 179, 6: 255, 7: 256, 8: 90, 9: 255}},

            
            'Shelf'       :   {'Default': {1: 85, 2: full_ang1-145, 3: 145, 4: 190, 5: full_ang2-190, 6: 180, 7:135, 8:180, 9:255},
                                'Target' : {1: 111.31702715038168, 2: 177.14316797592414, 3: 183.85683202407586, 4: 191.04324289760845, 5: 169.95675710239155, 6: 179.45891861047554, 7: 170, 8: 180, 9: 255},  
                                'Chair'  : {1: 38, 2: 250, 3: full_ang1-250, 4: 117, 5: full_ang2-117, 6: 273, 7: 204, 8: 89, 9: 255},
                        },

            'Floor'       :   {'Default': {1: 175, 2: 193, 3: 168, 4: 191, 5: 170, 6: 178, 7: 156, 8: 187, 9: 255},
                                'Target' : {1: 182.7964620728535, 2: 101.20365743496313, 3: 259.79634256503687, 4: 188.74850042860834, 5: 172.25149957139166, 6: 179.92160867784926, 7: 156, 8: 187, 9: 255},
                                'Chair'  : {1: 38, 2: 250, 3: full_ang1-250, 4: 117, 5: full_ang2-117, 6: 273, 7: 204, 8: 89, 9: 255}},

             'Before_Destruct' : {1: 84, 2: full_ang1-75, 3: 75,  4: 90, 5: full_ang2-90, 6: 180, 7:135, 8:180, 9:255},
                      }
        } 

    return Robot

class RoboticArm:
    
    def __init__ (self, CMD_dict, COM_ID = 'COM5', PROTOCOL_VERSION = 2.0):
        
        self.CMD              = CMD_dict['Real']['CMD']
        self.priority         = CMD_dict['Real']['Priority']
        self.home_position    = CMD_dict['Real']['Home']
        self.drinking_position    = CMD_dict['Real']['Drinking']['Default']

        self.before_destruct_position = CMD_dict['Real']['Before_Destruct']

        self.initialize(COM_ID, PROTOCOL_VERSION)
        
    def initialize(self, COM_ID, PROTOCOL_VERSION):  

        self.portHandler   = PortHandler  (COM_ID)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")

        # Broadcast ping the Dynamixel
        ENGINES_list, COMM_result = self.packetHandler.broadcastPing(self.portHandler)
        if COMM_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(COMM_result))
        self.ENGINES = []
        print("Detected Engines :")
        for engine in ENGINES_list:
            print("[ID:%03d] model version : %d | firmware version : %d" % 
                  (engine, ENGINES_list.get(engine)[0], ENGINES_list.get(engine)[1]))           
            if ENGINES_list.get(engine)[0] > 1000: # Checking for identified engines (rather then controllers)
                self.ENGINES.append(engine)

        # Set port baudrate    
        print('Setting baud rate to: 1 Mbps')
        for ID in self.ENGINES:
            self.send_single_cmd(ID, 
                          self.CMD['Baud Rate']['Address'], 
                          self.CMD['Baud Rate']['Value'][1000000])
        

        # Initializing mode and constraints
        self.reset_state()
        
        #Initializing syncronized actuation of coupled joints     
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler, self.packetHandler, self.CMD['Goal Position']['Address'], 4) # 4 for full CMD
        self.groupSyncWrite.clearParam()


        


     
    def reset_state (self):
    
        # Setting operation mode to position (default; getting into home position)
        # This command has to be excuted first. Changing modes reset default values.       
        print('Releasing torque')
        self.release_torque()
        
        print('Setting operation mode to: position')
        for ID in self.ENGINES:
            
            if ID == self.ENGINES[-1]:
                self.send_single_cmd(ID, 
                               self.CMD['Operating mode']['Address'], 
                               self.CMD['Operating mode']['Value']['Torque'])
            
            else:
            
                self.send_single_cmd(ID, 
                               self.CMD['Operating mode']['Address'], 
                               self.CMD['Operating mode']['Value']['Position'])
        
        # Limiting PWM
        print('Limiting Goal PWM to: {}%'.format(100*(self.CMD['Goal PWM']['Value']/885)))
        for ID in self.ENGINES:
            self.send_full_cmd(ID, 
                               self.CMD['Goal PWM']['Address'], 
                               self.CMD['Goal PWM']['Value'])


        # Set Current
        print('Set Goal Current to: {}%'.format(100*(self.CMD['Goal Current']['Value']/400))) # 400 = Limit Turqe
        for ID in self.ENGINES:
            self.send_full_cmd(ID, 
                               self.CMD['Goal Current']['Address'], 
                               self.CMD['Goal Current']['Value'])

        # Limiting velocity
        print('Limiting velocity to: {}%'.format(100*(self.CMD['Velocity Limit']['Value']/1023)))
        for ID in self.ENGINES:
            self.send_full_cmd(ID, 
                               self.CMD['Velocity Limit']['Address'], 
                               self.CMD['Velocity Limit']['Value'])

        # Limiting torque
        print('Limiting torque to: {}%'.format(100*(self.CMD['Limit torque']['Value']/1193)))
        for ID in self.ENGINES:
            self.send_half_cmd(ID, 
                               self.CMD['Limit torque']['Address'], 
                               self.CMD['Limit torque']['Value'])


        print('Setting Profile Velocity')
        for ID in self.ENGINES:
            self.send_full_cmd(ID, 
                        self.CMD['Profile Velocity']['Address'], 
                        self.CMD['Profile Velocity']['Value'])

        
        print('Setting Profile Acceleration')
        for ID in self.ENGINES:
            self.send_full_cmd(ID, 
                        self.CMD['Profile Acceleration']['Address'], 
                        self.CMD['Profile Acceleration']['Value'])

        print('Setting Shadow to: Disable')
        for ID in self.ENGINES:
            self.send_single_cmd(ID, 
                               self.CMD['Secondary(Shadow) ID']['Address'], 
                               self.CMD['Secondary(Shadow) ID']['Value']['Disable'])
               
     

    def change_current_gripper(self, v):
        self.send_half_cmd(self.ENGINES[-1], 
                               self.CMD['Goal Current']['Address'], 
                               v)
        
    
    def go_home (self) : # 0
        self.enable_torque()
        print("Setting home position")
        self.set_position_by_priority(self.home_position)

    
    def go_drinking (self) : # 1
        self.enable_torque()
        print("Setting drinking position")
        self.set_position_by_priority(self.drinking_position)

    def go_floor (self) : # 2
        self.enable_torque()
        print("Setting drinking position")
        self.set_position_by_priority(self.floor_position)

    def go_shelf (self) : # 3
        self.enable_torque()
        print("Setting drinking position")
        self.set_position_by_priority(self.shelf_position)
    
    
    def reboot (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            if DXL_ID < 12: # Assuming less than 12 engines' arm. 
                dxl_comm_result, dxl_error = self.packetHandler.reboot(self.portHandler, DXL_ID)
                if dxl_comm_result != COMM_SUCCESS:
                    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                elif dxl_error != 0:
                    print("%s" % self.packetHandler.getRxPacketError(dxl_error))
       
    def enable_torque (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            self.send_single_cmd(DXL_ID, 
                          self.CMD['Torque Enable']['Address'], 
                          self.CMD['Torque Enable']['Value']['ON'])  
    
    def release_torque (self, IDs = 'all'):
        
        if IDs == 'all':
            IDs = self.ENGINES
        
        for DXL_ID in IDs:
            self.send_single_cmd(DXL_ID, 
                          self.CMD['Torque Enable']['Address'], 
                          self.CMD['Torque Enable']['Value']['OFF'])
    
    def play_by_priority (self, sequance, delay, mode='position'):
        
        if mode == 'position':
            for seq in sequance:
                self.set_position_by_priority(seq)
                time.sleep(delay)
                
        elif mode == 'torque':
            
            targets_TDs = []
            for seq in sequance:
                for ID in seq:
                    if ID not in targets_TDs:
                        targets_TDs.append(ID)
            # Setting targets to torque control mode
            print('Setting operation mode of engines {} to: torque'.format(targets_TDs))
            for ID in targets_TDs:
                self.release_torque(targets_TDs) # Before changing mode, torque have to be released
                self.send_single_cmd(ID, 
                                   self.CMD['Operating mode']['Address'], 
                                   self.CMD['Operating mode']['Value']['Torque'])
                self.enable_torque(targets_TDs)
            
            for seq in sequance:
                self.set_torque(seq)
                time.sleep(delay)
        else:
            print('Working mode is not recognized. Supported modes: position / torque')
            
    def set_torque (self, torque_dict):
        
        for IDs in self.priority:

            for ID in IDs:
                
                if ID not in torque_dict:
                    continue
                
                target_torque = round(torque_dict[ID])
                print('Setting {} to {}'.format(ID, target_torque))
                self.send_half_cmd(ID, 
                                   self.CMD['Goal torque']['Address'], 
                                   target_torque)
                
    def watch_for_execution(self, ID, target):
        
        watchdog    = time.time()
        lapsed_time = 0

        while True:
            lapsed_time = time.time() - watchdog
            current_position = self.get_position(ID)
            error = round(target-current_position)
            print('{} Deviation is {}'.format(ID, error))
            con = abs(round(target-current_position)) < 500
            if con:
                return True
            if lapsed_time > 2.5:
                print('watch dog executed. ID: {} is at {} instead of {}'.format(ID, current_position, target))

                # Set target to current position. 
                self.send_full_cmd(ID, 
                    self.CMD['Goal Position']['Address'], 
                    self.get_position(ID))
                return False
            
    
    def set_position(self, position_dict):
        
        for ID in position_dict:

            target_position = position_dict[ID]
            target_position = int(round(target_position * 11.375))
            target_position_byte_array = [DXL_LOBYTE(DXL_LOWORD(target_position)), 
                                          DXL_HIBYTE(DXL_LOWORD(target_position)), 
                                          DXL_LOBYTE(DXL_HIWORD(target_position)), 
                                          DXL_HIBYTE(DXL_HIWORD(target_position))]
            self.groupSyncWrite.addParam(ID, target_position_byte_array)
                
        print('Setting position at {}'.format(position_dict))
        self.groupSyncWrite.txPacket()
        
    #    for ID in position_dict:
    #        self.watch_for_execution(ID, round(position_dict[ID] * 11.375)) 
        
        self.groupSyncWrite.clearParam()
    
    def set_position_by_priority (self, position_dict):

        for IDs in self.priority:
            
            if len(IDs) > 1: # Synchronized actuation
                
                for ID in IDs:
                    target_position = round(position_dict[ID] * 11.375)
                    target_position_byte_array = [DXL_LOBYTE(DXL_LOWORD(target_position)), 
                                                  DXL_HIBYTE(DXL_LOWORD(target_position)), 
                                                  DXL_LOBYTE(DXL_HIWORD(target_position)), 
                                                  DXL_HIBYTE(DXL_HIWORD(target_position))]
                    self.groupSyncWrite.addParam(ID, target_position_byte_array)
                print('Setting {} to {}'.format(IDs, target_position))
                self.groupSyncWrite.txPacket()
                
                for ID in IDs:
                    self.watch_for_execution(ID, target_position) 

            else:
                ID = IDs[0]
                
                if ID not in position_dict:
                    continue
                
                target_position = round(position_dict[ID])
                if target_position not in self.CMD['Ranges'][ID]:
                    print('{} is not in range. Engine {} is constrained to {}'.format(
                        position_dict[ID], ID, self.CMD['Ranges'][ID]))
                    continue

                # Multiplying angle by 11.375 to convert to register value
                target_position = round(target_position * 11.375)
                print('Setting {} to {}'.format(ID, target_position))

                self.send_full_cmd(ID, 
                              self.CMD['Goal Position']['Address'], 
                              target_position)
            
                self.watch_for_execution(ID, target_position) 
    
    def get_position(self, ID):
        
        dxl_present_position, dxl_comm_result, dxl_error = self.packetHandler.read4ByteTxRx(
            self.portHandler, ID, self.CMD['Present Position']['Address'])
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        return dxl_present_position
    
    def destruct(self):
        self.reboot()
        self.portHandler.closePort()
       
    def send_full_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        else:
            print("[ID:%03d] CMD executed successfully" % ID)
    
    def send_half_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        else:
            print("[ID:%03d] CMD executed successfully" % ID)
    
    def send_single_cmd(self, ID, adr, val):
        
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
                self.portHandler, ID, adr, val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            print('ID: {}, Address: {}, value: {}'.format(ID, adr, val))
        else:
            print("[ID:%03d] CMD executed successfully" % ID)
    
