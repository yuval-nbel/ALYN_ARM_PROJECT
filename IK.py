"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphic Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 15.9.2020
"""

import sympy as sp
import numpy as np


class viper300:
    """ Describe the Viperx200 6DOF robotic arm by Trossen Robotic
    
    The class provides the properties, transformation matrices and jacobian of the ViperX 300.
    The arm is described in: https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx
    """
    
    def __init__ (self):
        
        # Robots' joints
        self.n_joints = 5
        self.q0 = sp.Symbol('q0') 
        self.q1 = sp.Symbol('q1') 
        self.q2 = sp.Symbol('q2') 
        self.q3 = sp.Symbol('q3')
        self.q4 = sp.Symbol('q4')
        
        # length of the robots' links
        self.l1 = (127-9)      * 1e-3
        self.l2 = (427-127)    * 1e-3
        self.l3 = (60)         * 1e-3
        self.l4 = (253-60)     * 1e-3
        self.l5 = (359-253)    * 1e-3
        self.l6 = (567-359)    * 1e-3
               
        # Calculate the transformation matrix for base to EE in operational space
        self.Tx = self.calculate_Tx()[0].subs([('l1', self.l1), 
                                           ('l2', self.l2), 
                                           ('l3', self.l3), 
                                           ('l4', self.l4), 
                                           ('l5', self.l5),
                                           ('l6', self.l6)])

        

        self.T = self.calculate_Tx()[2].subs([('l1', self.l1), 
                                           ('l2', self.l2), 
                                           ('l3', self.l3), 
                                           ('l4', self.l4), 
                                           ('l5', self.l5),
                                           ('l6', self.l6)])

        self.J_o = self.calculate_Tx()[1]

        # Calculate the Jacobian matrix for the EE
        self.J = self.calculate_J().subs([('l1', self.l1), 
                                          ('l2', self.l2), 
                                          ('l3', self.l3), 
                                          ('l4', self.l4), 
                                          ('l5', self.l5), 
                                          ('l6', self.l6)])

        
    
    def calculate_Tx(self):
        """ Calculate the transformation matrix for base to EE in operational space """
        
        q0 = self.q0
        q1 = self.q1
        q2 = self.q2
        q3 = self.q3
        q4 = self.q4
        
        l1 = sp.Symbol('l1')
        l2 = sp.Symbol('l2')
        l3 = sp.Symbol('l3')
        l4 = sp.Symbol('l4')
        l5 = sp.Symbol('l5')
        l6 = sp.Symbol('l6')
      
        # Rotate around the z axis, connected the world's ref axis to the arm's rotating base  
        T01 = sp.Matrix([[sp.cos(q0),  -sp.sin(q0), 0, 0],
                         [sp.sin(q0),  sp.cos(q0),  0, 0],
                         [0,           0,           1, 0],
                         [0,           0,           0, 1]])

        T01_o = sp.Matrix([[1,  0,  0, 0],
                           [0,  1,  0, 0],
                           [0,  0,  1, 0],
                           [0,  0,  0, 1]])
        
        # Rotate around the x axis
        T12 = sp.Matrix([[1, 0,          0,           0 ],
                         [0, sp.cos(q1), -sp.sin(q1), 0 ],
                         [0, sp.sin(q1), sp.cos(q1),  l1],
                         [0, 0.        , 0,           1 ]])

        T12_o = sp.Matrix([[1,  0,  0, 0],
                           [0,  1,  0, 0],
                           [0,  0,  1, 0],
                           [0,  0,  0, 1]])
        
        # Rotate around the x axis
        T23 = sp.Matrix([[1, 0,          0,           0 ],
                         [0, sp.cos(q2), -sp.sin(q2), l3],
                         [0, sp.sin(q2), sp.cos(q2),  l2],
                         [0, 0.        , 0,           1 ]])
        
        T23_o = sp.Matrix([[1,  0,  0,   0],
                           [0,  1,  0,  0],
                           [0,  0,  1,  0],
                           [0,  0,  0,  1]])

        # Rotate around the y axis
        T34 = sp.Matrix([[sp.cos(q3), 0, sp.sin(q3), 0 ],
                         [0,          1, 0,          l4],
                         [-sp.sin(q3),0, sp.cos(q3), 0 ],
                         [0,          0, 0,          1 ]])

        T34_o = sp.Matrix([[1,  0,  0, 0],
                           [0,  1,  0, 0],
                           [0,  0,  1, 0],
                           [0,  0,  0, 1]])
        
        # Rotate around the x axis
        T45 = sp.Matrix([[1, 0,          0,           0 ],
                         [0, sp.cos(q4), -sp.sin(q4), l5],
                         [0, sp.sin(q4), sp.cos(q4),  0],
                         [0, 0.        , 0,           1 ]])

        T45_o = sp.Matrix([[1,  0,  0, 0],
                           [0,  1,  0, 0],
                           [0,  0,  1, 0],
                           [0,  0,  0, 1]])

        # Translate to the end effector
        T56 = sp.Matrix([[1, 0, 0, 0],
                         [0, 1, 0, l6],
                         [0, 0, 1, 0 ],
                         [0, 0, 0, 1 ]])

        T = T01 * T01_o * T12 * T12_o * T23 * T23_o * T34 * T34_o * T45 * T45_o * T56
        x = sp.Matrix([0, 0, 0, 1])
        Tx = T * x
        
        kx = sp.Matrix([1,0,0])
        ky = sp.Matrix([0,1,0])
        kz = sp.Matrix([0,0,1])

        o1 = (T01 *T01_o)[:3,:3] * kz
        o2 = (T01 * T01_o * T12 * T12_o )[:3,:3] * kx
        o3 = (T01 * T01_o * T12 * T12_o * T23 * T23_o)[:3,:3] * kx
        o4 = (T01 * T01_o * T12 * T12_o * T23 * T23_o * T34 * T34_o)[:3,:3] * ky
        o5 = (T01 * T01_o * T12 * T12_o * T23 * T23_o * T34 * T34_o * T45 * T45_o)[:3,:3] * kx

        J_orientation = [o1, o2, o3, o4, o5]

        return Tx, J_orientation, T
    
    def calculate_J(self):
        """ Calculate the Jacobian matrix for the EE """

        q = [self.q0, self.q1, self.q2, self.q3, self.q4]
        J = sp.Matrix.ones(6, 5)
        J_o = self.J_o
        for i in range(3):     # x, y, z
            for j in range(5): # Five joints
                # Differentiate and simplify
                J[i, j] = sp.simplify(self.T[i].diff(q[j])) 
                J[i+3, j] = J_o[j][i]
                
        return J
    
    def get_xyz_symbolic(self, q):
        """ Calculate EE location in operational space by solving the for Tx symbolically """
        
        return np.array(self.T.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')

    def calc_J_symbolic(self, q): 
        """ Calculate the jacobian symbolically """
        return np.array(self.J.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')
    
    def get_xyz_numeric(self, q):
        """ Calculate EE location in operational space by solving the for Tx numerically
        
        Equation was derived symbolically and was then written here manually.
        Nuerical evaluation works faster then symbolically. 
        """
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        sin = np.sin
        cos = np.cos

        T0 = 0.208*((sin(q0)*sin(q1)*cos(q2) + sin(q0)*sin(q2)*cos(q1))*cos(q3) + sin(q3)*cos(q0))*sin(q4) + \
            0.208*(sin(q0)*sin(q1)*sin(q2) - sin(q0)*cos(q1)*cos(q2))*cos(q4) + 0.299*sin(q0)*sin(q1)*sin(q2) + \
            0.3*sin(q0)*sin(q1) - 0.299*sin(q0)*cos(q1)*cos(q2) - 0.06*sin(q0)*cos(q1)

        
        T1 = 0.208*((-sin(q1)*cos(q0)*cos(q2) - sin(q2)*cos(q0)*cos(q1))*cos(q3) + sin(q0)*sin(q3))*sin(q4) + \
            0.208*(-sin(q1)*sin(q2)*cos(q0) + cos(q0)*cos(q1)*cos(q2))*cos(q4) - 0.299*sin(q1)*sin(q2)*cos(q0) - \
            0.3*sin(q1)*cos(q0) + 0.299*cos(q0)*cos(q1)*cos(q2) + 0.06*cos(q0)*cos(q1)
        

        T2 = 0.208*(-sin(q1)*sin(q2) + cos(q1)*cos(q2))*sin(q4)*cos(q3) + 0.208*(sin(q1)*cos(q2) + \
            sin(q2)*cos(q1))*cos(q4) + 0.299*sin(q1)*cos(q2) + 0.06*sin(q1) + 0.299*sin(q2)*cos(q1) + 0.3*cos(q1) + 0.118

        T3 = 1

        return np.array([[T0],
                         [T1],
                         [T2], 
                         [T3]], dtype='float')
    

    
    def calculate_R(self, q): # no o - v2
        """ Calculate EE location in operational space by solving the for Tx numerically
        
        Equation was derived symbolically and was then written here manually.
        Nuerical evaluation works faster then symbolically. 
        """
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        sin = np.sin
        cos = np.cos

        
        T0 = -(sin(q0)*sin(q1)*cos(q2) + sin(q0)*sin(q2)*cos(q1))*sin(q3) + cos(q0)*cos(q3)


        
        T1 = ((sin(q0)*sin(q1)*cos(q2) + sin(q0)*sin(q2)*cos(q1))*cos(q3) + sin(q3)*cos(q0))*sin(q4) + \
            (sin(q0)*sin(q1)*sin(q2) - sin(q0)*cos(q1)*cos(q2))*cos(q4)

        
        T2 = ((sin(q0)*sin(q1)*cos(q2) + sin(q0)*sin(q2)*cos(q1))*cos(q3) + sin(q3)*cos(q0))*cos(q4) - \
            (sin(q0)*sin(q1)*sin(q2) - sin(q0)*cos(q1)*cos(q2))*sin(q4)

        T4 = -(-sin(q1)*cos(q0)*cos(q2) - sin(q2)*cos(q0)*cos(q1))*sin(q3) + sin(q0)*cos(q3)


        T5 = ((-sin(q1)*cos(q0)*cos(q2) - sin(q2)*cos(q0)*cos(q1))*cos(q3) + \
            sin(q0)*sin(q3))*sin(q4) + (-sin(q1)*sin(q2)*cos(q0) + cos(q0)*cos(q1)*cos(q2))*cos(q4)


        T6 = ((-sin(q1)*cos(q0)*cos(q2) - sin(q2)*cos(q0)*cos(q1))*cos(q3) + sin(q0)*sin(q3))*cos(q4) - \
            (-sin(q1)*sin(q2)*cos(q0) + cos(q0)*cos(q1)*cos(q2))*sin(q4)


        T8 = -(-sin(q1)*sin(q2) + cos(q1)*cos(q2))*sin(q3)


        T9 = (-sin(q1)*sin(q2) + cos(q1)*cos(q2))*sin(q4)*cos(q3) + (sin(q1)*cos(q2) + sin(q2)*cos(q1))*cos(q4)


        T10 = (-sin(q1)*sin(q2) + cos(q1)*cos(q2))*cos(q3)*cos(q4) - (sin(q1)*cos(q2) + sin(q2)*cos(q1))*sin(q4)


        
        

        return np.array([[T0,  T1,  T2],
                    [T4,  T5,  T6],
                    [T8,  T9,  T10],], dtype='float')
 

    def calc_J_numeric(self, q):
        """ Calculate the Jacobian for q numerically
         
         Equation was derived symbolically and was then written here manually.
         Nuerical evaluation works faster then symbolically. 
         """
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        
        sin = np.sin
        cos = np.cos

        # position
        J0 = -0.208*sin(q0)*sin(q3)*sin(q4) + 0.3*sin(q1)*cos(q0) + 0.208*sin(q4)*sin(q1 + q2)*cos(q0)*cos(q3) - 0.06*cos(q0)*cos(q1) - \
            0.208*cos(q0)*cos(q4)*cos(q1 + q2) - 0.299*cos(q0)*cos(q1 + q2)


        J1 = (0.06*sin(q1) + 0.208*sin(q4)*cos(q3)*cos(q1 + q2) + 0.208*sin(q1 + q2)*cos(q4) + 0.299*sin(q1 + q2) + 0.3*cos(q1))*sin(q0)

        J2 = (0.208*sin(q4)*cos(q3)*cos(q1 + q2) + 0.208*sin(q1 + q2)*cos(q4) + 0.299*sin(q1 + q2))*sin(q0)
        
        J3 = -0.208*(sin(q0)*sin(q3)*sin(q1 + q2) - cos(q0)*cos(q3))*sin(q4)
        
        J4 = 0.208*(sin(q0)*sin(q1 + q2)*cos(q3) + sin(q3)*cos(q0))*cos(q4) + 0.208*sin(q0)*sin(q4)*cos(q1 + q2)
        
        J5 = 0.3*sin(q0)*sin(q1) + 0.208*sin(q0)*sin(q4)*sin(q1 + q2)*cos(q3) - 0.06*sin(q0)*cos(q1) - \
            0.208*sin(q0)*cos(q4)*cos(q1 + q2) - 0.299*sin(q0)*cos(q1 + q2) + 0.208*sin(q3)*sin(q4)*cos(q0)

        J6 = -(0.06*sin(q1) + 0.208*sin(q4)*cos(q3)*cos(q1 + q2) + 0.208*sin(q1 + q2)*cos(q4) + 0.299*sin(q1 + q2) + 0.3*cos(q1))*cos(q0)
       
        J7 = -(0.208*sin(q4)*cos(q3)*cos(q1 + q2) + 0.208*sin(q1 + q2)*cos(q4) + 0.299*sin(q1 + q2))*cos(q0)
        
        J8 = 0.208*(sin(q0)*cos(q3) + sin(q3)*sin(q1 + q2)*cos(q0))*sin(q4)
        
        J9 = 0.208*(sin(q0)*sin(q3) - sin(q1 + q2)*cos(q0)*cos(q3))*cos(q4) - 0.208*sin(q4)*cos(q0)*cos(q1 + q2)

        J10 = 0
        
        J11 = -0.3*sin(q1) - 0.208*sin(q4)*sin(q1 + q2)*cos(q3) + 0.06*cos(q1) + 0.208*cos(q4)*cos(q1 + q2) + 0.299*cos(q1 + q2)
        
        J12 = -0.208*sin(q4)*sin(q1 + q2)*cos(q3) + 0.208*cos(q4)*cos(q1 + q2) + 0.299*cos(q1 + q2)
        
        J13 = -0.208*sin(q3)*sin(q4)*cos(q1 + q2)
        
        J14 = -0.208*sin(q4)*sin(q1 + q2) + 0.208*cos(q3)*cos(q4)*cos(q1 + q2)  


        # oriantetion
        J15=0
        J16=cos(q0)
        J17=cos(q0)
        J18=sin(q0)*sin(q1)*sin(q2) - sin(q0)*cos(q1)*cos(q2)
        J19=-(sin(q0)*sin(q1)*cos(q2) + sin(q0)*sin(q2)*cos(q1))*sin(q3) + cos(q0)*cos(q3)


        J20=0
        J21=sin(q0)
        J22=sin(q0)
        J23=-sin(q1)*sin(q2)*cos(q0) + cos(q0)*cos(q1)*cos(q2)
        J24=-(-sin(q1)*cos(q0)*cos(q2) - sin(q2)*cos(q0)*cos(q1))*sin(q3) + sin(q0)*cos(q3)

        J25=1
        J26=0
        J27=0
        J28=sin(q1)*cos(q2) + sin(q2)*cos(q1)
        J29=-(-sin(q1)*sin(q2) + cos(q1)*cos(q2))*sin(q3)

        
        return np.array([[J0,  J1,  J2,  J3,  J4],
                         [J5,  J6,  J7,  J8,  J9],
                         [J10, J11, J12, J13, J14],
                         [J15,  J16,  J17,  J18,  J19],
                         [J20,  J21,  J22,  J23,  J24],
                         [J25, J26, J27, J28, J29]], dtype='float')

    
