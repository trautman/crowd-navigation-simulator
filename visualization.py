import matplotlib.pyplot as plt
import numpy as np
from dwa_controller import DWAController
from brne_controller import BRNEController


def visualize_dwa(state, goal, obstacles, controller):
    """
    Visualize DWA planning.
    - Plots all sampled trajectories and highlights the optimal path.
    """
    # Compute candidate trajectories and their scores
    v_min, v_max, w_min, w_max = controller.calc_dynamic_window(state)
    vs = np.linspace(v_min, v_max, controller.cfg['v_samples'])
    ws = np.linspace(w_min, w_max, controller.cfg['w_samples'])
    trajs, scores = [], []
    for v in vs:
        for w in ws:
            traj = controller.predict_trajectory(state, v, w)
            controller.cfg['current_v'] = v
            score = controller.evaluate_trajectory(traj, goal, obstacles)
            trajs.append(traj)
            scores.append(score)
    # Select best trajectory
    best_idx = int(np.argmax(scores))
    best_traj = trajs[best_idx]

    # Plotting
    plt.figure()
    # Plot obstacles
    if obstacles:
        ox, oy = zip(*obstacles)
        plt.scatter(ox, oy, c='red', label='Obstacles')
    # Plot robot start
    px, py = state[0], state[1]
    plt.scatter(px, py, c='blue', label='Robot')
    # Plot all candidates (unpack list of (x,y) tuples)
    for traj in trajs:
        xs, ys = zip(*traj)
        plt.plot(xs, ys, linewidth=0.5, alpha=0.3)
    # Highlight best path (unpack)
    xs, ys = zip(*best_traj)
    plt.plot(xs, ys, linewidth=2, label='Optimal DWA path')
    plt.legend()
    plt.axis('equal')
    plt.show()



def visualize_brne(state, goal, ped_list, controller):
    """
    Visualize BRNE planning.
    - Shows GP samples for robot and pedestrians, and NE trajectories for each pedestrian.
    """
    # Run control to populate controller.last_* attributes
    v, w = controller.control(state, goal, ped_list)
    robot_samples = controller.last_robot_samples  # shape (N, tsteps, 2)
    ped_samples = controller.last_ped_samples      # list of (N, tsteps, 2)
    ped_ne = controller.last_ped_trajs             # list of (tsteps, 2)

    plt.figure()
    # Robot GP samples
    for traj in robot_samples:
        xs, ys = traj[:,0], traj[:,1]
        plt.plot(xs, ys, linewidth=0.5, alpha=0.3, color='gray')
    # Robot start
    plt.scatter([state[0]], [state[1]], c='blue', label='Robot')
    # Pedestrian GP samples
    for samples in ped_samples:
        for traj in samples:
            xs, ys = traj[:,0], traj[:,1]
            plt.plot(xs, ys, linewidth=0.5, alpha=0.3)
    # NE trajectories
    for i, ne in enumerate(ped_ne):
        xs, ys = ne[:,0], ne[:,1]
        plt.plot(xs, ys, linewidth=2, label=f'Ped {i} NE')
    plt.legend()
    plt.axis('equal')
    plt.show()

def draw_gp_samples(ax, state, obstacles, controller, num_samples=20, horizon=5):
    """
    Draw num_samples GP trajectories of length “horizon” on the given Axes ax.
    Assumes controller has a method `sample_gp(state, obstacles, num_samples, horizon)`
    that returns an array of shape (num_samples, horizon, 2).
    """
    # get your GP trajectories
    samples = controller.sample_gp(state, obstacles,
                                   num_samples=num_samples,
                                   horizon=horizon)
    # plot each sample
    for traj in samples:
        ax.plot(traj[:, 0], traj[:, 1],
                linewidth=1,
                alpha=0.3)
