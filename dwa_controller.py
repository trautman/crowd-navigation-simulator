import numpy as np
import math

class DWAController:
    """
    Dynamic Window Approach controller.
    """

    def __init__(self, config, time_step):
        """
        config: dict with keys
          max_speed, max_yaw_rate, max_accel, max_dyaw_rate,
          dt, predict_time, v_samples, w_samples,
          heading_weight, distance_weight, velocity_weight
        time_step: simulation time step (float)
        """
        self.cfg = config
        self.dt = time_step

    def calc_dynamic_window(self, state):
        """
        state: [x, y, theta, v, omega]
        returns (v_min, v_max, w_min, w_max)
        """
        v, w = state[3], state[4]
        vmax       = self.cfg['max_speed']
        wmax       = self.cfg['max_yaw_rate']
        # velocity window
        v_min = max(0.0, v - self.cfg['max_accel'] * self.dt)
        v_max = min(vmax, v + self.cfg['max_accel'] * self.dt)
        # yaw rate window
        w_min = max(-wmax, w - self.cfg['max_dyaw_rate'] * self.dt)
        w_max = min(wmax, w + self.cfg['max_dyaw_rate'] * self.dt)
        return v_min, v_max, w_min, w_max

    def predict_trajectory(self, state, v, w):
        """
        Simulate forward for predict_time, return list of (x,y).
        """
        traj = []
        x, y, th = state[0], state[1], state[2]
        t = 0.0
        while t < self.cfg['predict_time']:
            x += v * math.cos(th) * self.dt
            y += v * math.sin(th) * self.dt
            th += w * self.dt
            traj.append((x, y))
            t += self.dt
        return traj

    def evaluate_trajectory(self, traj, goal, obstacles):
        """
        Score a trajectory by:
         - heading: closeness to goal at end,
         - clearance: min distance to obstacles,
         - velocity: normalized speed.
        """
        # heading
        last = np.array(traj[-1])
        error = np.linalg.norm(goal - last)
        heading_score = (self.cfg['predict_time'] - error) / self.cfg['predict_time']
        # clearance

        # POINT MASS CALCULATION WRONG
        # dmin = float('inf')
        # for ox, oy in obstacles:
        #     for px, py in traj:
        #         dmin = min(dmin, np.hypot(ox-px, oy-py))

        # get the sum of radii so we treat both robot & ped as circles
        sep = self.cfg.get('robot_radius', 0.0) + self.cfg.get('agent_radius', 0.0)
        dmin = float('inf')
        for px, py in traj:
            for ox, oy in obstacles:
                # center-to-center minus required clearance
                dist = np.hypot(ox-px, oy-py) - sep
                if dist < 0:
                    # collision!
                    return -np.inf
                dmin = min(dmin, dist)

        clearance_score = min(dmin, self.cfg['predict_time']) / self.cfg['predict_time']
        # velocity
        # assume config['current_v'] set before call
        velocity_score = self.cfg.get('current_v', 0.0) / self.cfg['max_speed']
        # weighted sum
        return ( self.cfg['heading_weight'] * heading_score
               + self.cfg['distance_weight'] * clearance_score
               + self.cfg['velocity_weight'] * velocity_score )

    def control(self, state, goal, obstacles):
        """
        Return best (v, w) from sampling dynamic window.
        """
        v_min, v_max, w_min, w_max = self.calc_dynamic_window(state)
        best, best_score = (0.0, 0.0), -float('inf')
        for v in np.linspace(v_min, v_max, self.cfg['v_samples']):
            for w in np.linspace(w_min, w_max, self.cfg['w_samples']):
                traj = self.predict_trajectory(state, v, w)
                # pass current v for velocity score
                self.cfg['current_v'] = v
                score = self.evaluate_trajectory(traj, goal, obstacles)
                if score > best_score:
                    best_score, best = score, (v, w)
        return best

    def motion(self, state, control):
        """
        Integrate state = [x,y,theta,v,w] by one time_step.
        """
        x, y, th, _, _ = state
        v, w = control
        x += v * math.cos(th) * self.dt
        y += v * math.sin(th) * self.dt
        th += w * self.dt
        return [x, y, th, v, w]
