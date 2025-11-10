import control as ct    
import numpy as np
import matplotlib.pyplot as plt

m, c, k = 1.0, 0.2, 5.0  # mass, damping coefficient, spring constant

A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [-2*k/m, k/m, -c/m, 0], [k/m, -2*k/m, 0, -c/m]])
B = np.array([[0], [0], [0], [k/m]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
D = 0

sys = ct.ss(A, B, C, D, outputs=['q1', 'q2'], name='Coupled Mass-Spring-Damper System')
print(sys)

response = ct.initial_response(sys, X0=[1, 0, 0, 0])
# cplt = response.plot()

t = response.time
x = response.states
# plt.plot(t, x[0], 'b', t, x[1], 'r')
# plt.legend(['$x_1$', '$x_2$'])
# plt.xlim(0, 50)
# plt.ylabel('States')
# plt.xlabel('Time [s]')
# plt.title("Initial response from $x_1 = 1$, $x_2 = 0$")
# plt.show()

#######
# Kinematic Robot

def vehicle_update(t, x, u, params):
    a = params.get('refoffset', 1.5)        # offset to vehicle reference point
    b = params.get('wheelbase', 3.)         # vehicle wheelbase
    maxsteer = params.get('maxsteer', 0.5)  # maximum steering angle [rad]

    delta = np.clip(u[1], -maxsteer, maxsteer)  # steering angle
    alpha = np.arctan2(a * np.tan(delta), b)  # slip angle

    return np.array([u[0] * np.cos(x[2] + alpha),
                     u[0] * np.sin(x[2] + alpha),
                     u[0] * np.sin(alpha) / a]) 

def vehicle_output(t, x, u, params):
    return x

vehicle_params={'refoffset': 1.5, 'wheelbase': 3.0, 'maxsteer': 0.5}

vehicle = ct.nlsys(vehicle_update, vehicle_output, states=3, name='Kinematic Vehicle', inputs=['v','delta'], outputs=['x','y','theta'], params=vehicle_params)
timepts = np.linspace(0, 10, 1000)

U = [
    10*np.ones_like(timepts),  # constant speed input
    0.1*np.sin(2*np.pi*timepts)    # sinusoidal steering input
]

response = ct.input_output_response(vehicle, timepts, U)
time, outputs, inputs = response.time, response.outputs, response.inputs

fig, ax = plt.subplots(2, 1)

# Plot the results in the xy plane
ax[0].plot(outputs[0], outputs[1])
ax[0].set_xlabel("$x$ [m]")
ax[0].set_ylabel("$y$ [m]")

# Plot the inputs
ax[1].plot(timepts, U[0])
ax[1].set_ylim(0, 12)
ax[1].set_xlabel("Time $t$ [s]")
ax[1].set_ylabel("Velocity $v$ [m/s]")
ax[1].yaxis.label.set_color('blue')

rightax = ax[1].twinx()       # Create an axis in the right
rightax.plot(timepts, U[1], color='red')
rightax.set_ylim(None, 0.5)
rightax.set_ylabel(r"Steering angle $\phi$ [rad]")
rightax.yaxis.label.set_color('red')

fig.tight_layout()
plt.show()