import numpy as np
from casadi import *
import math
import matplotlib.pyplot as plt

def partial_measure(state):
    dx = state[0] @ cos(state[2]) @ sin(state[1])
    dy = state[0] @ cos(state[2]) @ cos(state[1])
    dz = state[0] @ sin(state[2])

    r = sqrt(dx**2 + dy**2 + dz**2) + 1e-8  # Adding a small value to avoid division by zero

    H_i = vertcat(
        horzcat(-dx / r, -dy / r, -dz / r),
        horzcat(-dz * dx / (r * sqrt(dx**2 + dy**2) + 1e-8), -dz * dy / (r * sqrt(dx**2 + dy**2) + 1e-8), sqrt(dx**2 + dy**2) / (r**2 + 1e-8)),
        horzcat(dy / (dx**2 + dy**2 + 1e-8), -dx / (dx**2 + dy**2 + 1e-8), 0)
    )
    return H_i

def Distance(state):
    dx = state[0] @ cos(state[2]) @ sin(state[1])
    dy = state[0] @ cos(state[2]) @ cos(state[1])
    dz = state[0] @ sin(state[2])

    return sqrt(dx**2 + dy**2 + dz**2)


def information_matrix(robot):
    P = np.eye(3)
    R = np.eye(3)
    R[0,0] = 4e-2
    R[1,1] = 4e-1
    R[2,2] = 3e-1
    num = 3
    temp_s = []

    for i in range(num):
        H = partial_measure(robot[:, i])
        inv_matrix = inv(R )
        temp_s.append((-H.T) @ inv_matrix @ (-H))

    alpha = 0
    for i in range(num):
        alpha += trace(temp_s[i])

    s = np.zeros([3, 3])
    for i in range(num):
        s += alpha * temp_s[i]
    return 1/trace(s)  # Return the trace of the matrix to get a scalar value
def agent_distance(uav1,uav2):
    dx = uav1[0] - uav2[0] 
    dy = uav1[1] - uav2[1]
    dz = uav1[2] - uav2[2]

    return  sqrt(dx**2 + dy**2 + dz**2)
def coordinate(optimal):
    target = [0,0,0]
    uavs = []    
    for i in range(3):
        uav = horzcat(target[0] + optimal[0,i]*cos(optimal[2,i])*sin(optimal[1,i]) , target[1] + optimal[0,i]*cos(optimal[2,i])*cos(optimal[1,i]) , target[2] + optimal[0,i]*sin(optimal[2,i]))
        uavs = vertcat(uavs , uav)
    return uavs
def plot_position_3D(optimal):
    fig = plt.figure()
    target = [0,0,10]
    uavs = []
    for i in range(3):
        uav = [target[0] + optimal[0,i]*cos(optimal[2,i])*sin(optimal[1,i]) , target[1] + optimal[0,i]*cos(optimal[2,i])*cos(optimal[1,i]) , target[2] + optimal[0,i]*sin(optimal[2,i])]
        uavs = horzcat(uavs , uav)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(uavs[0,:], uavs[1,:], uavs[2,:], c = 'r', label='GT Pose target')
    ax.scatter(target[0], target[1], target[2], c = 'g', marker = 'o', label='EST Pose t')
    # ax.scatter(1,1,1, c = 'b', marker = 'o', label='test')

    print(uavs)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    plt.title('target 3D Poses')
    plt.show()
opti = casadi.Opti()
num = 3

# Define variables within the Opti context
x = opti.variable(3, 3)  # 3 UAVs each with 3 coordinates (d, alpha, beta)
# t = opti.variable(3)  # Target position with 3 coordinates (t1, t2, t3)

obj = information_matrix(x)
opti.minimize(obj)

# Relax the constraints for feasibility
min_distance = 3.0
target_distance = 3.0

# for i in range(num):
#     for j in range(num):
#         if i != j:
#             opti.subject_to(Distance(x[:, i]) >= min_distance)
for i in range(num):
    opti.subject_to(x[0, i] > 3.0)
    opti.subject_to(x[0, i] < 5.0)
    opti.subject_to(x[1, i] < pi)
    opti.subject_to(x[2, i] < pi)
    opti.subject_to(x[1, i] > -pi)
    opti.subject_to(x[2, i] > -pi)
uavs = coordinate(x)
opti.subject_to(agent_distance(uavs[0,:],uavs[1,:]) > 3.0)
opti.subject_to(agent_distance(uavs[0,:],uavs[2,:]) > 3.0)
opti.subject_to(agent_distance(uavs[1,:],uavs[2,:]) > 3.0)
# Provide an initial guess for the variables
initial_x = np.array([[3, 20, 1], [2, 3, 6], [1, 3, 6]])
initial_t = np.array([1, 1, 1])
opti.set_initial(x, initial_x)
# opti.set_initial(t, initial_t)

# Debug initial values
print("Initial x:", initial_x)
# print("Initial t:", initial_t)


opti.solver('ipopt')

try:
    sol = opti.solve()
    print("Optimal x:", sol.value(x))
    # print("Optimal t:", sol.value(t))

    plot_position_3D(sol.value(x))
except RuntimeError as e:
    print("Solver failed with message:", str(e))
    print("Debug values for x:", opti.debug.value(x))
    # print("Debug values for t:", opti.debug.value(t))

