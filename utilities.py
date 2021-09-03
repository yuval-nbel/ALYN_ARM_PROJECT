import math
import numpy as np
import nengo

def euler_from_quaternion(quaternion):

    _EPS = np.finfo(float).eps * 4.0

    def quaternion_matrix(quaternion):

        q = np.array(quaternion, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array(
            [[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
             [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
             [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )

    def euler_from_matrix(matrix):

        _AXES2TUPLE = (2, 1, 0, 1) # in "rxyz" axis
        _NEXT_AXIS  = [1, 2, 0, 1] # axis sequences for Euler angles

        firstaxis, parity, repetition, frame = _AXES2TUPLE

        i = firstaxis
        j = _NEXT_AXIS[i + parity]
        k = _NEXT_AXIS[i - parity + 1]

        M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
        if repetition:
            sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
            if sy > _EPS:
                ax = math.atan2(M[i, j], M[i, k])
                ay = math.atan2(sy, M[i, i])
                az = math.atan2(M[j, i], -M[k, i])
            else:
                ax = math.atan2(-M[j, k], M[j, j])
                ay = math.atan2(sy, M[i, i])
                az = 0.0
        else:
            cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
            if cy > _EPS:
                ax = math.atan2(M[k, j], M[k, k])
                ay = math.atan2(-M[k, i], cy)
                az = math.atan2(M[j, i], M[i, i])
            else:
                ax = math.atan2(-M[j, k], M[j, j])
                ay = math.atan2(-M[k, i], cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az

    return euler_from_matrix(quaternion_matrix(quaternion))

def calc_orientation_forces(target_abg, R_e):
    u_task_orientation = np.zeros(3)

    # get rotation matrix for the target orientation
    R_d = euler_matrix(
        target_abg[0], target_abg[1], target_abg[2], 
    )[:3, :3]
    R_ed = np.dot(R_e.T, R_d)  # eq 24
    q_ed = unit_vector(
        quaternion_from_matrix(R_ed)
    )
    u_task_orientation = -1 * np.dot(R_e, q_ed[1:])  # eq 34

    return u_task_orientation




def euler_matrix(ai, aj, ak):
        """Return homogeneous rotation matrix from Euler angles and axis sequence.
        ai, aj, ak : Euler's roll, pitch and yaw angles
        axes : One of 24 axis sequences as string or encoded tuple
        """
        _AXES2TUPLE = (2, 1, 0, 1) # in "rxyz" axis
    #    _AXES2TUPLE = (0, 0, 0, 0) # in "sxyz" axis
        _NEXT_AXIS  = [1, 2, 0, 1] # axis sequences for Euler angles
        
        firstaxis, parity, repetition, frame = _AXES2TUPLE


        i = firstaxis
        j = _NEXT_AXIS[i + parity]
        k = _NEXT_AXIS[i - parity + 1]

        if frame:
            ai, ak = ak, ai
        if parity:
            ai, aj, ak = -ai, -aj, -ak

        si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
        ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        M = np.identity(4)
        if repetition:
            M[i, i] = cj
            M[i, j] = sj * si
            M[i, k] = sj * ci
            M[j, i] = sj * sk
            M[j, j] = -cj * ss + cc
            M[j, k] = -cj * cs - sc
            M[k, i] = -sj * ck
            M[k, j] = cj * sc + cs
            M[k, k] = cj * cc - ss
        else:
            M[i, i] = cj * ck
            M[i, j] = sj * sc - cs
            M[i, k] = sj * cc + ss
            M[j, i] = cj * sk
            M[j, j] = sj * ss + cc
            M[j, k] = sj * cs - sc
            M[k, i] = -sj
            M[k, j] = cj * si
            M[k, k] = cj * ci
        return M

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ],
        dtype=np.float64,
    )

def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q

def quaternion_from_euler(ai, aj, ak, axes="rxyz"):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    """
    _AXES2TUPLE = (2, 1, 0, 1) # in "rxyz" axis
 #   _AXES2TUPLE = (0, 0, 0, 0) # in "sxyz" axis
    _NEXT_AXIS  = [1, 2, 0, 1] # axis sequences for Euler angles
        
    firstaxis, parity, repetition, frame = _AXES2TUPLE


    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0

    return q




def get_intercepts(n_neurons, dimensions):

    triangular = np.random.triangular(left=0.35, 
                                      mode=0.45, 
                                      right=0.55, 
                                      size=n_neurons)
                                      
    intercepts = nengo.dists.CosineSimilarity(dimensions + 2).ppf(1 - triangular)
    return intercepts

def calc_T(q):

    c0 = np.cos(q[0])
    c1 = np.cos(q[1])
    c2 = np.cos(q[2])
    c3 = np.cos(q[3])
    c4 = np.cos(q[4])
    
    s0 = np.sin(q[0])
    s1 = np.sin(q[1])
    s2 = np.sin(q[2])
    s3 = np.sin(q[3])
    s4 = np.sin(q[4])
    
    return np.array([[0.208*((-s1*c0*c2 - s2*c0*c1)*c3 + s0*s3)*s4 + 
                      0.208*(-s1*s2*c0 + c0*c1*c2)*c4 - 0.299*s1*s2*c0 - 
                      0.3*s1*c0 + 0.299*c0*c1*c2 + 0.06*c0*c1],
                     [0.208*(-s1*s2 + c1*c2)*s4*c3 + 
                      0.208*(s1*c2 + s2*c1)*c4 + 
                      0.299*s1*c2 + 0.06*s1 + 0.299*s2*c1 + 0.3*c1 + 0.118],
                     [0.208*((s0*s1*c2 + s0*s2*c1)*c3 + s3*c0)*s4 + 
                      0.208*(s0*s1*s2 - s0*c1*c2)*c4 + 0.299*s0*s1*s2 + 
                      0.3*s0*s1 - 0.299*s0*c1*c2 - 0.06*s0*c1]], dtype='float')

def calc_J(q):

    c0 = np.cos(q[0])
    c1 = np.cos(q[1])
    c2 = np.cos(q[2])
    c3 = np.cos(q[3])
    c4 = np.cos(q[4])
    
    s0 = np.sin(q[0])
    s1 = np.sin(q[1])
    s2 = np.sin(q[2])
    s3 = np.sin(q[3])
    s4 = np.sin(q[4])

    s12  = np.sin(q[1] + q[2])
    c12  = np.cos(q[1] + q[2])

    
    return np.array([[0.3*s0*s1 + 0.208*s0*s4*s12*c3 - 0.06*s0*c1 - 0.208*s0*c4*c12 - 
                      0.299*s0*c12 + 0.208*s3*s4*c0, -(0.06*s1 + 0.208*s4*c3*c12 + 
                      0.208*s12*c4 + 0.299*s12 + 0.3*c1)*c0,
                      -(0.208*s4*c3*c12 + 0.208*s12*c4 + 0.299*s12)*c0,
                      0.208*(s0*c3 + s3*s12*c0)*s4, 0.208*(s0*s3 - s12*c0*c3)*c4 - 
                      0.208*s4*c0*c12],
                     [0,-0.3*s1 - 0.208*s4*s12*c3 + 0.06*c1 + 0.208*c4*c12 + 0.299*c12,
                      -0.208*s4*s12*c3 + 0.208*c4*c12 + 0.299*c12,-0.208*s3*s4*c12,
                      -0.208*s4*s12 + 0.208*c3*c4*c12],
                     [-0.208*s0*s3*s4 + 0.3*s1*c0 + 0.208*s4*s12*c0*c3 - 0.06*c0*c1 - 
                      0.208*c0*c4*c12 -0.299*c0*c12,(0.06*s1 + 0.208*s4*c3*c12 + 0.208*s12*c4 + 
                      0.299*s12 + 0.3*c1)*s0,(0.208*s4*c3*c12 + 0.208*s12*c4 + 
                      0.299*s12)*s0, -0.208*(s0*s3*s12 - c0*c3)*s4,
                      0.208*(s0*s12*c3 + s3*c0)*c4 + 0.208*s0*s4*c12]], dtype='float')

def calc_T_O(q):
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

def calc_J_O(q):
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


def calculate_R(q): # no o - v2
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
 
