import numpy as np
import math

from brne import (
    get_Lmat_nb,
    mvn_sample_normal,
    get_ulist_essemble,
    traj_sim_essemble,
    brne_nav,
)

class BRNEController:
    """
    BRNEController implements the BRNE joint‐reciprocal‐navigation algorithm.
    """

    def __init__(self, cfg, dt):
        self.cfg = cfg
        self.dt  = dt

        # how strongly to scale the pedestrian GP draws
        self.ped_sample_scale = cfg.get("ped_sample_scale", 1.0)

        # placeholders for debugging/visualization
        self.last_robot_samples = None
        self.last_ped_samples   = []
        self.last_ped_trajs     = []
        self.last_W             = None

    def sample_gp(self, state, obstacles, num_samples=20, horizon=5):
        """
        Return GP samples for either pedestrians or the robot.
        You’ll need to hook into whatever GP you built in brne.py—
        for example, use your brne_nav() internals or a saved self.gp_model.
        """
        # 1) build your test inputs X_star = future state grid of shape (horizon, feature_dim)
        # 2) call your GP: mu, cov = self.gp_model.predict(X_star, return_cov=True)
        # 3) draw samples: raw = np.random.multivariate_normal(mu, cov, size=num_samples)
        # 4) reshape to (num_samples, horizon, 2) and return:
        return raw.reshape(num_samples, horizon, 2)


    def control(self, state, goal, ped_list):
        # 1) Sample GPs
        robot_trajs, ped_trajs = self.sample_gps(state, ped_list, goal)
        self.last_robot_samples = robot_trajs
        self.last_ped_samples   = ped_trajs
        self.last_ped_trajs     = []

        # If no pedestrians in view, just head straight to goal
        if not ped_list:
            px, py, th = state[0], state[1], state[2]
            desired    = math.atan2(goal[1]-py, goal[0]-px)
            err        = (desired - th + math.pi) % (2*math.pi) - math.pi
            w          = np.clip(err, -self.cfg["max_yaw_rate"], self.cfg["max_yaw_rate"])
            v          = float(self.cfg["nominal_vel"])
            return v, w

        cfg    = self.cfg
        N      = cfg["num_samples"]
        tsteps = cfg["gp"]["tsteps"]

        # 2) Build nominal control ensemble
        u_nom = np.tile([cfg["nominal_vel"], 0.0], (tsteps, 1))  # (tsteps,2)
        u_ens = get_ulist_essemble(
            u_nom,
            cfg["max_speed"],
            cfg["max_yaw_rate"],
            N
        )  # (tsteps, N, 2)

        # 3) Simulate robot rollouts
        init_st     = np.tile(state[:3], (N,1)).T    # (3, N)
        robot_rolls = traj_sim_essemble(init_st, u_ens, self.dt)
        # robot_rolls.shape == (tsteps, 3, N)

        # 4) Assemble joint trajectories
        num_agents = 1 + len(ped_trajs)
        xtraj = np.zeros((num_agents * N, tsteps))
        ytraj = np.zeros((num_agents * N, tsteps))

        # 4a) robot block
        xtraj[0:N, :] = robot_rolls[:, 0, :].T  # (N, tsteps)
        ytraj[0:N, :] = robot_rolls[:, 1, :].T

        # 4b) pedestrian blocks
        for j, ped in enumerate(ped_trajs, start=1):
            s, e = j * N, (j + 1) * N
            # ped.shape == (N, tsteps, 2)
            xtraj[s:e, :] = ped[:, :, 0]
            ytraj[s:e, :] = ped[:, :, 1]

        # 5) Run the BRNE optimizer
        W = brne_nav(
            xtraj, ytraj,
            num_agents, tsteps, N,
            cfg["gp"]["cost_a1"],
            cfg["gp"]["cost_a2"],
            cfg["gp"]["cost_a3"],
            self.ped_sample_scale,
            cfg.get("corridor_y_min"),
            cfg.get("corridor_y_max"),
        )
        self.last_W = W

        # 6) Extract each pedestrian's NE trajectory
        for pi in range(len(ped_trajs)):
            s, e = (pi + 1) * N, (pi + 2) * N
            rows = xtraj[s:e, :]  # shape (N, tsteps)
            cols = ytraj[s:e, :]

            w_i_raw = W[0, s:e]   # ideally length N
            # ─── GUARD against bad weight length ───
            if w_i_raw.shape[0] != N or w_i_raw.sum() == 0.0:
                # skip this pedestrian if no valid weights
                continue
            w_i = w_i_raw

            x_ne = (rows * w_i[:, None]).sum(axis=0) / w_i.sum()
            y_ne = (cols * w_i[:, None]).sum(axis=0) / w_i.sum()
            self.last_ped_trajs.append(np.stack([x_ne, y_ne], axis=1))

        # 7) First‐step robot command from NE weights
        w0 = W[0, :N]
        vs = u_ens[0, :, 0]
        ws = u_ens[0, :, 1]
        v  = float(np.dot(w0, vs) / w0.sum())
        w  = float(np.dot(w0, ws) / w0.sum())

        return v, w

    def motion(self, state, control):
        x, y, θ = state[0], state[1], state[2]
        v, w    = control
        x  += v * math.cos(θ) * self.dt
        y  += v * math.sin(θ) * self.dt
        θ  += w * self.dt
        return [x, y, θ, v, w]

    def sample_gps(self, state, ped_list, robot_goal):
        """
        Returns:
          - robot_trajs: np.array of shape (N, tsteps, 2)
          - ped_trajs:   list of np.arrays, each shape (N, tsteps, 2)
        """
        cfg     = self.cfg
        N       = cfg["num_samples"]
        tsteps  = cfg["gp"]["tsteps"]
        horizon = cfg["gp"].get("horizon", tsteps * self.dt)
        times   = np.arange(tsteps) * self.dt

        # pull and coerce obs_noise to float64
        obs_noise = np.array(cfg["gp"].get("obs_noise", [1e-4, 1e-4]), dtype=float)

        # build covariance matrix
        Lmat, _ = get_Lmat_nb(
            np.array([0.0, horizon]),
            times,
            obs_noise,
            cfg["gp"]["kernel_a1"],
            cfg["gp"]["kernel_a2"],
        )
        frac = np.clip(times / horizon, 0.0, 1.0)

        # — robot GP toward goal —
        px, py = state[0], state[1]
        gx, gy = robot_goal
        vec    = np.array([gx - px, gy - py])
        dist   = np.linalg.norm(vec)
        dir_uv = vec / dist if dist > 1e-6 else np.zeros(2)
        meas_r = np.array([px, py]) + horizon * dir_uv

        mean_x = px + frac * (meas_r[0] - px)
        mean_y = py + frac * (meas_r[1] - py)

        x_dev = mvn_sample_normal(N, tsteps, Lmat)
        y_dev = mvn_sample_normal(N, tsteps, Lmat)
        robot_trajs = np.stack([x_dev + mean_x, y_dev + mean_y], axis=2)

        # — pedestrian GPs, scaled —
        ped_trajs = []
        scale = self.ped_sample_scale
        for ped in ped_list:
            px, py = ped['pos']
            gx, gy = ped['goal']
            vec    = np.array([gx - px, gy - py])
            dist   = np.linalg.norm(vec)
            dir_uv = vec / dist if dist > 1e-6 else np.zeros(2)
            meas_p = np.array([px, py]) + horizon * dir_uv

            mx = px + frac * (meas_p[0] - px)
            my = py + frac * (meas_p[1] - py)

            xd = mvn_sample_normal(N, tsteps, Lmat) * scale
            yd = mvn_sample_normal(N, tsteps, Lmat) * scale

            ped_trajs.append(np.stack([xd + mx, yd + my], axis=2))

        return robot_trajs, ped_trajs
