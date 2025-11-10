import control as ct    
import numpy as np
import matplotlib.pyplot as plt

uav_params={
    'mass': 1.5,                                # mass of the UAV [kg]
    'inertia': np.diag([0.03, 0.03, 0.05]),     # inertia matrix [kg*m^2]
    'arm_length': 0.2,                          # distance from center to motor [m]
    'max_thrust': 150.0,                         # maximum thrust per motor [N]
    'max_torque': 1.0,                          # maximum torque per motor [N*m]
    'ct': 50.0,                                  # thrust coefficient
    'cm' : 1.0                                  # moment coefficient
}


def rotationX(phi):
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])
def rotationY(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])
def rotationZ(psi):
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def transformEulderDot(phi, theta, psi):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    t_theta = np.tan(theta)
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    return np.array([[1, 0, -s_theta],
                     [0, c_phi, -s_phi],
                     [0, s_phi / c_theta, c_phi / c_theta]])

def uav_update(t, x, u, params):
    m = params.get('mass')      
    I = params.get('inertia')         
    R = params.get('arm_length')  
    max_thrust = params.get('max_thrust')
    max_torque = params.get('max_torque')
    ct = params.get('ct')
    cm = params.get('cm')
    g = 9.81

    px = x[0]
    py= x[1]
    pz= x[2]
    phi = x[3]
    theta = x[4]
    psi = x[5]
    vx = x[6]
    vy = x[7]
    vz = x[8]
    p = x[9]
    q = x[10]
    r = x[11]

    u1, u2, u3, u4 = np.clip(u, 0, max_thrust) 

    motor_positions = np.array([
        [0.14,  0.14, 0.0],   # front-right
        [-0.14, -0.14, 0.0],  # rear-left
        [-0.14,  0.14, 0.0],  # front-left
        [0.14, -0.14, 0.0]    # rear-right
    ])
    spin_dir = np.array([1, -1, 1, -1])  # CW or CCW

    # Calculate Euler derivative
    eul_dot = transformEulderDot(phi, theta, psi) @ np.array([p, q, r])

    # Calculate position derivative
    vel_B = np.array([vx, vy, vz])
    vel_I = rotationZ(psi) @ rotationY(theta) @ rotationX(phi) @ vel_B

    # Calculate total thrust 
    Tm_B =  motor_positions[0]*u1 + \
            motor_positions[1]*u2 + \
            motor_positions[2]*u3 + \
            motor_positions[3]*u4
    Tg_I = np.array([0, 0, -m*g])

    Tm_I = rotationZ(psi) @ rotationY(theta) @ rotationX(phi) @ Tm_B
    # Total force in inertial frame
    T_I = Tm_I + Tg_I

    # Total moment in body frame
    Mm_B = np.zeros(3)
    for i in range(4):
        thrust = np.array([0, 0, u[i]])
        moment_arm = motor_positions[i]
        Mm_B += np.cross(moment_arm, thrust) + np.array([0, 0, spin_dir[i]*cm*u[i]])

    # print(np.shape(x))

    return np.array([vel_I, eul_dot, T_I, Mm_B]).flatten()

def uav_output(t, x, u, params):
    return np.array(x)


drone = ct.nlsys(uav_update, uav_output, states=12, name='Quadrotor UAV', inputs=['u1','u2','u3','u4'], outputs=['px','py','pz','phi','theta','psi','vx','vy','vz','p','q','r'], params=uav_params)

# timepts = np.linspace(0, 10, 1000)


# response = ct.input_output_response(drone, timepts, U)
# time, outputs, inputs = response.time, response.outputs, response.inputs


# --- Parametry symulacji ---

dt = 0.01
T = 5.0
steps = int(T / dt)
time = np.linspace(0, T, steps)
reference_eul = np.array([0.0, 0.0, 0.0]) # desired roll, pitch, yaw
reference_pos = np.array([0.0, 0.0, -2.0]) # desired x, y, z position

Kp, Ki, Kd = 105.0, 0.4, 10.0  # PID gains for altitude control
integral = np.array([0.0, 0.0, 0.0])
prev_error_eul = np.array([0.0, 0.0, 0.0])
prev_error_pos = np.array([0.0, 0.0, 0.0])

ROLL = 3
PITCH = 4
YAW = 5

# Inicjalizacja stanu 
x = np.zeros(12)
roll_history = []
pitch_history = []
yaw_history = []
px_history = []
py_history = []
pz_history = [] 
u_history = []

# --- Pętla symulacji ---
for t in time:
    roll = x[ROLL]
    pitch = x[PITCH]
    yaw = x[YAW]
    px = x[0]
    py = x[1]
    pz = x[2]
    vx = x[6]
    vy = x[7]
    vz = x[8]

    # error_eul = reference_eul - np.array([roll, pitch, yaw])
    # integral += error_eul * dt
    # derivative = (error_eul - prev_error_eul) / dt
    # prev_error_eul = error_eul

    error_pos = reference_pos - np.array([px, py, pz])
    integral += error_pos * dt
    derivative = (error_pos - prev_error_pos) / dt
    prev_error_pos = error_pos 

    # PIDR
    # pidr = Kp * error_eul[0] + Ki * integral[0] + Kd * derivative[0]

    # PIDT
    pidt = Kp * error_pos[2] + Ki * integral[2] + Kd * derivative[2]

    # każdy silnik dostaje tyle samo
    u = np.clip(np.ones(4) * pidt / 4, 0, uav_params['max_thrust'])

    # print(x)
    # print(u)

    # RK4 integrator
    k1 = uav_update(t, x, u, uav_params)
    k2 = uav_update(t + dt/2, x + dt/2*k1, u, uav_params)
    k3 = uav_update(t + dt/2, x + dt/2*k2, u, uav_params)
    k4 = uav_update(t + dt, x + dt*k3, u, uav_params)
    x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    pz_history.append(pz)
    u_history.append(pidt)

    print(f"Czas: {t:.2f} s, Wysokość: {pz:.2f} m, Prędkość: {vz:.2f} m/s, Sterowanie: {pidt:.2f} N")


# --- Wizualizacja ---
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.plot(time, pz_history, label='z (wysokość)')
plt.axhline(reference_pos[2], color='r', linestyle='--', label='z_ref')
plt.xlabel("Czas [s]")
plt.ylabel("Wysokość [m]")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(time, u_history, label='PID output (całkowity ciąg)')
plt.xlabel("Czas [s]")
plt.ylabel("Sterowanie")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


