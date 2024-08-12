import rosbag
import matplotlib.pyplot as plt
import numpy as np
def extract_data(bag, topics):
    timestamps = []
    RMSE_p = []
    RMSE_v = []
    GT_poses = []
    est_poses = []
    det_p= []
    for topic, msg, t in bag.read_messages(topics):
    # Extract relevant data from the message
        timestamps.append(msg.header.stamp.to_sec())
        RMSE_p.append(msg.RMSE_p)
        RMSE_v.append(msg.RMSE_v)
        GT_poses.append(msg.GT_pose)
        est_poses.append(msg.est_pose)
        det_p.append(msg.det_p)
    return timestamps, GT_poses, est_poses, RMSE_p, RMSE_v, det_p



def plot_RMSE_p(timeStamps, RMSE_p, dataset_label):
    plt.figure(figsize=(10, 6))
    average_RMSE_p = sum(RMSE_p) / len(RMSE_p)  # Calculate average RMSE for position
    plt.plot(timeStamps, RMSE_p, label=f'Position RMSE for {dataset_label}')
    plt.axhline(y=average_RMSE_p, color='r', linestyle='--', label=f'Average RMSE: {average_RMSE_p:.3f}')  # Plot average RMSE line
    plt.text(timeStamps[int(len(timeStamps) / 10)], average_RMSE_p, f'Average RMSE: {average_RMSE_p:.3f}', color='red')  # Print average RMSE on plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position RMSE (m)')
    plt.title(f'Position RMSE for {dataset_label}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_RMSE_v(timeStamps, RMSE_v, dataset_label):
    plt.figure(figsize=(10, 6))
    average_RMSE_v = sum(RMSE_v) / len(RMSE_v)  # Calculate average RMSE for velocity
    plt.plot(timeStamps, RMSE_v, label=f'Velocity RMSE for {dataset_label}', color='orange')
    plt.axhline(y=average_RMSE_v, color='r', linestyle='--', label=f'Average RMSE: {average_RMSE_v:.3f}')  # Plot average RMSE line
    plt.text(timeStamps[int(len(timeStamps) / 10)], average_RMSE_v, f'Average RMSE: {average_RMSE_v:.3f}', color='red')  # Print average RMSE on plot
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity RMSE (m/s)')
    plt.title(f'Velocity RMSE for {dataset_label}')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_det_p(timeStamps, det_p, dataset_label):
    plt.figure(figsize = (10,6))
    average = sum(det_p)/len(det_p)
    # print(average)
    plt.plot(timeStamps, det_p, label=f'det p for {dataset_label}',color='orange')
    plt.axhline(y=average, color='r', linestyle='--',label=f'Average det_p: {average}')
    plt.text(timeStamps[int(len(timeStamps)/ 10)], average, f'Average : {average}', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('')
    plt.title(f'det p for {dataset_label}')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_combined_position_3D(GT_poses1, est_poses1,
                               GT_poses2, est_poses2,
                                GT_poses3, est_poses3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # First set of poses
    GT_x1 = [pose.position.x for pose in GT_poses1]
    GT_y1 = [pose.position.y for pose in GT_poses1]
    GT_z1 = [pose.position.z for pose in GT_poses1]
    est_x1 = [pose.position.x for pose in est_poses1]
    est_y1 = [pose.position.y for pose in est_poses1]
    est_z1 = [pose.position.z for pose in est_poses1]

    # Second set of poses
    GT_x2 = [pose.position.x for pose in GT_poses2]
    GT_y2 = [pose.position.y for pose in GT_poses2]
    GT_z2 = [pose.position.z for pose in GT_poses2]
    est_x2 = [pose.position.x for pose in est_poses2]
    est_y2 = [pose.position.y for pose in est_poses2]
    est_z2 = [pose.position.z for pose in est_poses2]

    # Third set of poses
    GT_x3 = [pose.position.x for pose in GT_poses3]
    GT_y3 = [pose.position.y for pose in GT_poses3]
    GT_z3 = [pose.position.z for pose in GT_poses3]
    est_x3 = [pose.position.x for pose in est_poses3]
    est_y3 = [pose.position.y for pose in est_poses3]
    est_z3 = [pose.position.z for pose in est_poses3]

    # Plotting
    ax.plot(GT_x1, GT_y1, GT_z1, label='GT Pose 1')
    ax.scatter(est_x1, est_y1, est_z1,  marker = 'o', label='Est Pose 1')
    ax.plot(GT_x2, GT_y2, GT_z2, label='GT Pose 2')
    ax.scatter(est_x2, est_y2, est_z2,  marker = 'o', label='Est Pose 2')
    ax.plot(GT_x3, GT_y3, GT_z3, label='GT Pose 3')
    ax.scatter(est_x3, est_y3, est_z3,  marker = 'o', label='Est Pose 3')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    plt.title('Combined 3D Poses')
    plt.show()

def plot_target_position_3D(GT_posest, est_posest):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Target set of poses
    GT_xt = [pose.position.x for pose in GT_posest]
    GT_yt = [pose.position.y for pose in GT_posest]
    GT_zt = [pose.position.z for pose in GT_posest]
    est_xt = [pose.position.x for pose in est_posest]
    est_yt = [pose.position.y for pose in est_posest]
    est_zt = [pose.position.z for pose in est_posest]

    # Plotting
    ax.plot(GT_xt, GT_yt, GT_zt, label='GT Pose target')
    ax.scatter(est_xt, est_yt, est_zt, c = 'g', marker = 'o', label='EST Pose t')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    plt.title('target 3D Poses')
    # plt.savefig('target_position.png')
    plt.show()
def ploterror3D(GT_poses1, est_poses1,
                GT_poses2, est_poses2,
                GT_poses3, est_poses3,
                GT_posest, est_posest,
                new_bool):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # First set of poses
    GT_x1 = np.array([pose.position.x for pose in GT_poses1])
    GT_y1 = np.array([pose.position.y for pose in GT_poses1])
    GT_z1 = np.array([pose.position.z for pose in GT_poses1])
    est_x1 = np.array([pose.position.x for pose in est_poses1])
    est_y1 = np.array([pose.position.y for pose in est_poses1])
    est_z1 = np.array([pose.position.z for pose in est_poses1])

    # Second set of poses
    GT_x2 = np.array([pose.position.x for pose in GT_poses2])
    GT_y2 = np.array([pose.position.y for pose in GT_poses2])
    GT_z2 = np.array([pose.position.z for pose in GT_poses2])
    est_x2 = np.array([pose.position.x for pose in est_poses2])
    est_y2 = np.array([pose.position.y for pose in est_poses2])
    est_z2 = np.array([pose.position.z for pose in est_poses2])

    # Third set of poses
    GT_x3 = np.array([pose.position.x for pose in GT_poses3])
    GT_y3 = np.array([pose.position.y for pose in GT_poses3])
    GT_z3 = np.array([pose.position.z for pose in GT_poses3])
    est_x3 = np.array([pose.position.x for pose in est_poses3])
    est_y3 = np.array([pose.position.y for pose in est_poses3])
    est_z3 = np.array([pose.position.z for pose in est_poses3])

    #target
    GT_xt = np.array([pose.position.x for pose in GT_posest])
    GT_yt = np.array([pose.position.y for pose in GT_posest])
    GT_zt = np.array([pose.position.z for pose in GT_posest])
    est_xt = np.array([pose.position.x for pose in est_posest])
    est_yt = np.array([pose.position.y for pose in est_posest])
    est_zt = np.array([pose.position.z for pose in est_posest])    
    # Plotting
    ax.scatter(GT_x1[new_bool], GT_y1[new_bool], GT_z1[new_bool],c = 'r', s = 5, label='GT Pose 1')
    # ax.scatter(est_x1[new_bool], est_y1[new_bool], est_z1[new_bool], c = 'b', s = 5,  label='Est Pose 1')
    ax.scatter(GT_xt[new_bool], GT_yt[new_bool], GT_zt[new_bool],c = 'b', marker = '^', label='GT Pose t')
    ax.scatter(est_xt[new_bool], est_yt[new_bool], est_zt[new_bool],c = 'g', marker = '^', label='EST Pose t')
    ax.scatter(GT_x2[new_bool], GT_y2[new_bool], GT_z2[new_bool], c = 'y', s = 5, label='GT Pose 2')

    print(len(new_bool))
    # new_bool = np.append(new_bool, False)
    print(len(new_bool))

    # ax.scatter(est_x2[new_bool], est_y2[new_bool], est_z2[new_bool], c = 'r', s = 5, label='Est Pose 2')
    ax.scatter(GT_x3[new_bool], GT_y3[new_bool], GT_z3[new_bool], c = 'b', s = 5, label='GT Pose 3')
    # ax.scatter(est_x3[new_bool], est_y3[new_bool], est_z3[new_bool], c = 'r', s = 5 , label='Est Pose 3')


    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    plt.title('error 3D Poses')
    plt.show()    
def plot_imu(timeStamps, GT_poses, dataset_label):
    plt.figure(figsize = (10,6))

    GT_x = np.array([pose.orientation.x for pose in GT_poses])
    GT_y = np.array([pose.orientation.y for pose in GT_poses])
    GT_z = np.array([pose.orientation.z for pose in GT_poses])    

    plt.plot(timeStamps, GT_x, label=f'orientation x for {dataset_label}',color='orange')
    plt.plot(timeStamps, GT_y, label=f'orientation y for {dataset_label}',color='blue')
    # plt.plot(timeStamps, GT_z, label=f'orientation z for {dataset_label}',color='green')
    
    # plt.axhline(y=average, color='r', linestyle='--',label=f'Average det_p: {average}')
    # plt.text(timeStamps[int(len(timeStamps)/ 10)],  color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('')
    plt.title(f'orenitation for {dataset_label}')
    plt.legend()
    plt.grid(True)
    plt.show()
def plotFromBag(bag, name):
    E_x = []
    E_y = []
    E_z = []
    timestamps1, EIF_1_GTpose, EIF_1_Estpose, EIF_1_RMSE_p, EIF_1_RMSE_v , EIF_1_det_p  = extract_data(bag, '/typhoon_h480_1/SHEIF/Plot')
    timestamps2, EIF_2_GTpose, EIF_2_Estpose, EIF_2_RMSE_p, EIF_2_RMSE_v , EIF_2_det_p  = extract_data(bag, '/typhoon_h480_2/SHEIF/Plot')
    timestamps3, EIF_3_GTpose, EIF_3_Estpose, EIF_3_RMSE_p, EIF_3_RMSE_v , EIF_3_det_p  = extract_data(bag, '/typhoon_h480_3/SHEIF/Plot')
    timestampst, EIF_t_GTpose, EIF_t_Estpose, EIF_t_RMSE_p, EIF_t_RMSE_v , EIF_t_det_p = extract_data(bag, '/typhoon_h480_1/THEIF/Plot')
    # for (GTpose, Estpose) in zip(EIF_1_GTpose, EIF_1_Estpose):
        # E_x.append(abs(GTpose.position.x - Estpose.position.x))
        # E_y.append(abs(GTpose.position.y - Estpose.position.y))
        # E_z.append(abs(GTpose.position.z - Estpose.position.z))

    bag.close()
    
    # ploterror3D(EIF_1_GTpose, EIF_1_Estpose
    #             , EIF_2_GTpose, EIF_2_Estpose
    #             , EIF_3_GTpose, EIF_3_Estpose
    #             , EIF_t_GTpose, EIF_t_Estpose
    #             , new_bool)
    
    plot_combined_position_3D(EIF_1_GTpose, EIF_1_Estpose
                               , EIF_2_GTpose, EIF_2_Estpose
                               , EIF_3_GTpose, EIF_3_Estpose)
    
    plot_target_position_3D(EIF_t_GTpose, EIF_t_Estpose)

    # plot_RMSE_p(timestamps1, EIF_1_RMSE_p, "1")
    # plot_RMSE_v(timestamps1, EIF_1_RMSE_v, "1")
    # plot_RMSE_p(timestamps2, EIF_2_RMSE_p, "2")
    # plot_RMSE_v(timestamps2, EIF_2_RMSE_v, "2")
    # plot_RMSE_p(timestamps3, EIF_3_RMSE_p, "3")
    # plot_RMSE_v(timestamps3, EIF_3_RMSE_v, "3")
    plot_RMSE_p(timestampst, EIF_t_RMSE_p, "target")
    plot_RMSE_v(timestampst, EIF_t_RMSE_v, "target")
    # plot_imu(timestamps1,EIF_1_GTpose, "iris_1")
    # plot_imu(timestamps2,EIF_2_GTpose, "iris_2")
    # plot_imu(timestamps3,EIF_3_GTpose, "iris_3")
    
    plot_det_p(timestampst, EIF_t_det_p, "target")
    plot_det_p(timestamps1, EIF_1_det_p, "1")
    plot_det_p(timestamps2, EIF_2_det_p, "2")

def plot_combined_RMSE_p(RMSE_p1, label1, RMSE_p2, label2):
    plt.figure(figsize=(10, 6))
    
    # Normalize time steps: Start both from zero
    min_length = min(len(RMSE_p1), len(RMSE_p2))
    timeStamps1 = list(range(min_length))  # Creating a list from 0 to min_length
    timeStamps2 = list(range(min_length))

    # Cut the RMSE lists to match the new timestamps if necessary
    RMSE_p1 = RMSE_p1[:min_length]
    RMSE_p2 = RMSE_p2[:min_length]
    average_RMSE_p1 = sum(RMSE_p1) / len(RMSE_p1)
    average_RMSE_p2 = sum(RMSE_p2) / len(RMSE_p2)


    # Plot
    plt.plot(timeStamps1, RMSE_p1, label=f'Position RMSE for {label1}')
    plt.plot(timeStamps2, RMSE_p2, label=f'Position RMSE for {label2}')
    plt.axhline(y=average_RMSE_p1, color='b', linestyle='--', label=f'Average RMSE: {average_RMSE_p1:.3f} ({label1})')
    plt.axhline(y=average_RMSE_p2, color='r', linestyle='--', label=f'Average RMSE: {average_RMSE_p2:.3f} ({label2})')
    plt.xlabel('Normalized Time Steps')
    plt.ylabel('Position RMSE (m)')
    plt.title('Combined Position RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combined_RMSE_v(RMSE_v1, label1, RMSE_v2, label2):
    plt.figure(figsize=(10, 6))
    
    # Normalize time steps: Start both from zero
    min_length = min(len(RMSE_v1), len(RMSE_v2))
    timeStamps1 = list(range(min_length))  # Creating a list from 0 to min_length
    timeStamps2 = list(range(min_length))

    # Cut the RMSE lists to match the new timestamps if necessary
    RMSE_v1 = RMSE_v1[:min_length]
    RMSE_v2 = RMSE_v2[:min_length]
    average_RMSE_v1 = sum(RMSE_v1) / len(RMSE_v1)
    average_RMSE_v2 = sum(RMSE_v2) / len(RMSE_v2)

    # Plot
    plt.plot(timeStamps1, RMSE_v1, label=f'Velocity RMSE for {label1}')
    plt.plot(timeStamps2, RMSE_v2, label=f'Velocity RMSE for {label2}')
    plt.axhline(y=average_RMSE_v1, color='b', linestyle='--', label=f'Average RMSE: {average_RMSE_v1:.3f} ({label1})')
    plt.axhline(y=average_RMSE_v2, color='r', linestyle='--', label=f'Average RMSE: {average_RMSE_v2:.3f} ({label2})')
    plt.xlabel('Normalized Time Steps')
    plt.ylabel('Velocity RMSE (m)')
    plt.title('Velocity RMSE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combine_det_p( det_p, det_p2, dataset_label, dataset_label2):
    plt.figure(figsize = (10,6))
    min_length = min(len(det_p), len(det_p2))
    timeStamps = list(range(min_length))  # Creating a list from 0 to min_length



    # Cut the RMSE lists to match the new timestamps if necessary
    det_p = det_p[:min_length]
    det_p2 = det_p2[:min_length]

    average = sum(det_p)/len(det_p)
    average2 = sum(det_p2)/len(det_p2)  
    # print(average)
    plt.scatter(timeStamps, det_p, label=f'trace(p) for {dataset_label}',color='orange')
    plt.axhline(y=average, color='r', linestyle='--',label=f'Average trace(p): {average}')
    # plt.text(timeStamps[int(len(timeStamps)/ 10)], average, f'Average : {average}', color='red')
    plt.scatter(timeStamps, det_p2, label=f'trace(p) for {dataset_label2}',color='blue')
    plt.axhline(y=average2, color='g', linestyle='--',label=f'Average trace(p): {average2}')
    # plt.text(timeStamps[int(len(timeStamps)/ 10)], average, f'Average : {average2}', color='red')    
    plt.xlabel('Time (seconds)')
    plt.ylabel('')
    plt.title(f'target_trace(p)  (all uavs use gps)')
    plt.legend()
    plt.grid(True)
    plt.show()
def plotFromTwoBags(file1, file2, topic, label1, label2):
    # Open both bag files
    bag1 = rosbag.Bag(file1)
    bag2 = rosbag.Bag(file2)

    # Extract data from both bags
    _, _, _, _, _,p1 = extract_data(bag1, topic)
    _, _, _, _, _,p2 = extract_data(bag2, topic)

    # Close the bag files
    bag1.close()
    bag2.close()

    # Plot combined RMSE for position from both bags with normalized time steps
    # plot_combined_RMSE_p(RMSE_p1, label1, RMSE_p2, label2)
    # plot_combined_RMSE_v(RMSE_v1, label1, RMSE_v2, label2)
    plot_combine_det_p( p1, p2, label1, label2)
folder = '/home/py/eif_ws/src/state_estimation/bag/'

file1 = folder + 'gimbal.bag'
file2 = folder + 'worst.bag'
# file2 = folder + 'lidar.bag'
bag1 = rosbag.Bag(file1)
bag2 = rosbag.Bag(file2)
topic = '/typhoon_h480_2/THEIF/Plot'
plotFromTwoBags(file1, file2, topic, 'optimal', 'worst case')
# plotFromBag(bag1, 'THEIF, Only one neigbor robots has absolute position rate 5hz')