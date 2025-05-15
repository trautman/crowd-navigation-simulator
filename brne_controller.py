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
        # ── 0) remember robot’s start‐x for corridor bounds ─────────────
        if not hasattr(self, 'x0'):
            self.x0 = state[0]

        cfg    = self.cfg
        N      = cfg["num_samples"]
        tsteps = cfg["gp"]["tsteps"]
        dt     = self.dt

        # ── 1) OPEN‐SPACE: no pedestrians → simple arc toward goal ─────
        if not ped_list:
            px, py, th = state[0], state[1], state[2]
            gx, gy     = goal

            # compute yaw_os exactly like in ROS
            if th > 0.0:
                theta_a = th - np.pi/2
            else:
                theta_a = th + np.pi/2
            axis_vec  = np.array([np.cos(theta_a), np.sin(theta_a)])
            vec2goal  = np.array([gx-px, gy-py])
            dist2goal = np.linalg.norm(vec2goal)
            denom     = vec2goal @ vec2goal or 1.0
            proj_len  = (axis_vec @ vec2goal) / denom * dist2goal
            radius    = 0.5 * dist2goal / proj_len if proj_len else np.inf

            v_os   = float(cfg.get('open_space_velocity', cfg['nominal_vel']))
            yaw_os = (-v_os/radius) if th > 0 else (v_os/radius)

            # clip & return immediately
            v = np.clip(v_os,   -cfg['max_speed'],   cfg['max_speed'])
            w = np.clip(yaw_os, -cfg['max_yaw_rate'], cfg['max_yaw_rate'])
            return v, w

        # ── 2) CROWDED‐SPACE: pedestrians in FOV ─────────────────────────

        # 2.1) Sample GP rollouts
        robot_trajs, ped_trajs = self.sample_gps(state, ped_list, goal)
        self.last_robot_samples = robot_trajs
        self.last_ped_samples   = ped_trajs
        self.last_ped_trajs     = []

        # 2.2) Build nominal turn‐toward‐goal ensemble (ut)
        px, py, th = state[0], state[1], state[2]
        if th > 0.0:
            theta_a = th - np.pi/2
        else:
            theta_a = th + np.pi/2
        axis_vec  = np.array([np.cos(theta_a), np.sin(theta_a)])
        vec2goal  = np.array(goal) - np.array([px, py])
        dist2goal = np.linalg.norm(vec2goal)
        denom     = vec2goal @ vec2goal or 1.0
        proj_len  = (axis_vec @ vec2goal) / denom * dist2goal
        radius    = 0.5 * dist2goal / proj_len if proj_len else np.inf

        nominal_vel = cfg["nominal_vel"]
        if th > 0.0:
            ut = np.array([nominal_vel, -nominal_vel / radius])
        else:
            ut = np.array([nominal_vel,  nominal_vel / radius])
        u_ens = get_ulist_essemble(
            np.tile(ut, (tsteps, 1)),
            cfg["max_speed"],
            cfg["max_yaw_rate"],
            N
        )

        # 2.3) Simulate robot‐only rollouts → xtraj, ytraj
        init_st     = np.tile(state[:3], (N,1)).T
        robot_rolls = traj_sim_essemble(init_st, u_ens, dt)
        xtraj_robot = robot_rolls[:, 0, :].T  # shape (N, tsteps)
        ytraj_robot = robot_rolls[:, 1, :].T

        # 2.4) Assemble joint trajectories (robot + pedestrians)
        num_agents = 1 + len(ped_trajs)
        xtraj = np.zeros((num_agents * N, tsteps))
        ytraj = np.zeros((num_agents * N, tsteps))
        xtraj[0:N, :] = xtraj_robot
        ytraj[0:N, :] = ytraj_robot
        for j, ped in enumerate(ped_trajs, start=1):
            s, e = j*N, (j+1)*N
            xtraj[s:e, :] = ped[:, :, 0]
            ytraj[s:e, :] = ped[:, :, 1]

        # 2.5) Compute absolute corridor bounds in world‐x
        x_min = self.x0 + cfg["corridor_y_min"]
        x_max = self.x0 + cfg["corridor_y_max"]

        px, py, th = state[0], state[1], state[2]
    # if we’re outside the lateral bounds, steer back to corridor center
        if px < x_min or px > x_max:
            # centerline x
            x_center = self.x0 + 0.5*(cfg["corridor_y_min"] + cfg["corridor_y_max"])
            # purely lateral correction
            dx = x_center - px
            # zero forward component—just turn in place toward center
            desired = math.atan2(0.0, dx)  
            err     = (desired - th + math.pi) % (2*math.pi) - math.pi
            w_back  = np.clip(err, -cfg["max_yaw_rate"], cfg["max_yaw_rate"])
            v_back  = 0.0  # spin in place; or use nominal_vel if you want to drive laterally
            return v_back, w_back


        # 2.6) Call the NE optimizer with the correct num_agents
        W = brne_nav(
            xtraj, ytraj,
            num_agents, tsteps, N,
            cfg["gp"]["cost_a1"],
            cfg["gp"]["cost_a2"],
            cfg["gp"]["cost_a3"],
            self.ped_sample_scale,
            x_min, x_max
        )
        self.last_W = W

        # 2.7) Extract each pedestrian’s NE for visualization
        self.last_ped_trajs = []
        for pi in range(len(ped_trajs)):
            s, e = (pi+1)*N, (pi+2)*N
            w_i = W[0, s:e]
            if w_i.sum() > 0:
                rows = xtraj[s:e, :]
                cols = ytraj[s:e, :]
                x_ne = (w_i[:, None] * rows).sum(axis=0) / w_i.sum()
                y_ne = (w_i[:, None] * cols).sum(axis=0) / w_i.sum()
                self.last_ped_trajs.append(np.stack([x_ne, y_ne], axis=1))

        # 2.8) First‐step robot command from NE weights (with debug)
        w0    = W[0, :N]
        denom = w0.sum()
        vs    = u_ens[:,0,0]  # u_ens has shape (tsteps, N, 2)
        ws    = u_ens[:,0,1]  # but you want the first rollout step: use u_ens[0,:,*]
        vs    = u_ens[0,:,0]
        ws    = u_ens[0,:,1]

        # print(f"[DEBUG] w0 sum = {denom}, any NaNs? {np.isnan(denom)}", flush=True)

        if denom > 0 and not np.isnan(denom):
            v = float((w0 @ vs) / denom)
            w = float((w0 @ ws) / denom)
            # print(f"[DEBUG] used weighted NE, denom = {denom}", flush=True)
        else:
            v, w = ut
            # print(f"[DEBUG] fallback to ut, denom = {denom}", flush=True)

        # 2.9) Clip & return
        v = np.clip(v, -cfg['max_speed'],   cfg['max_speed'])
        w = np.clip(w, -cfg['max_yaw_rate'], cfg['max_yaw_rate'])
        return v, w



    # def control(self, state, goal, ped_list):
    #     # remember the very first x-position so we can build absolute corridor bounds
    #     if not hasattr(self, 'x0'):
    #         self.x0 = state[0]
    #         print("state 0", state[0])

    #     cfg    = self.cfg
    #     N      = cfg["num_samples"]
    #     tsteps = cfg["gp"]["tsteps"]
    #     dt     = self.dt



    #     if not ped_list:
    #         # open-space: no pedestrians → straight-to-goal at open_space_velocity
    #         px, py, th = state[0], state[1], state[2]
    #         gx, gy     = goal

    #         # compute yaw_os exactly like in ROS
    #         if th > 0.0:
    #             theta_a = th - np.pi/2
    #         else:
    #             theta_a = th + np.pi/2
    #         axis_vec = np.array([np.cos(theta_a), np.sin(theta_a)])
    #         vec2goal = np.array([gx-px, gy-py])
    #         dist2goal = np.linalg.norm(vec2goal)
    #         denom = vec2goal @ vec2goal or 1.0
    #         proj_len = (axis_vec @ vec2goal) / denom * dist2goal
    #         radius   = 0.5 * dist2goal / proj_len if proj_len else np.inf

    #         v_os = float(cfg.get('open_space_velocity', cfg['nominal_vel']))
    #         yaw_os = (-v_os/radius) if th>0 else (v_os/radius)

    #         # clip & return immediately
    #         return ( np.clip(v_os, -cfg['max_speed'],   cfg['max_speed']),
    #                  np.clip(yaw_os, -cfg['max_yaw_rate'], cfg['max_yaw_rate']) )




    #     # # ── OPEN-SPACE EARLY EXIT ───────────────────────────────
    #     # # If there are no pedestrians in FOV, build & optimize an
    #     # # ensemble around open_space_velocity and return immediately.
    #     # if not ped_list:
    #         # px, py, th = state[0], state[1], state[2]
    #         # gx, gy     = goal

    #         # # compute turning‐radius axis
    #         # if th > 0.0:
    #         #     theta_a = th - np.pi/2
    #         # else:
    #         #     theta_a = th + np.pi/2
    #         # axis_vec  = np.array([np.cos(theta_a), np.sin(theta_a)])
    #         # vec2goal  = np.array([gx - px, gy - py])
    #         # dist2goal = np.linalg.norm(vec2goal)
    #         # denom     = vec2goal @ vec2goal or 1.0
    #         # proj_len  = (axis_vec @ vec2goal) / denom * dist2goal
    #         # radius    = 0.5 * dist2goal / proj_len if proj_len else np.inf

    #         # # fetch open-space speed & yaw
    #         # v_os   = float(cfg.get('open_space_velocity', cfg['nominal_vel']))
    #         # yaw_os = (-v_os/radius) if th > 0 else (v_os/radius)

    #         # # build & sample the ensemble
    #         # nominal_cmds = np.tile([v_os, yaw_os], (tsteps, 1))
    #         # u_ens = get_ulist_essemble(
    #         #     nominal_cmds,
    #         #     cfg['max_speed'],
    #         #     cfg['max_yaw_rate'],
    #         #     N
    #         # )

    #         # # simulate & optimize exactly as in your crowded branch
    #         # init_st     = np.tile(state[:3], (N,1)).T
    #         # robot_rolls = traj_sim_essemble(init_st, u_ens, dt)
    #         # # (assemble xtraj/ytraj, call brne_nav to get W, just like below)
    #         # xtraj = np.zeros((N, tsteps)); ytraj = np.zeros((N, tsteps))
    #         # xtraj[:] = robot_rolls[:,0,:].T
    #         # ytraj[:] = robot_rolls[:,1,:].T
            

    #         #     # compute world-frame x-limits from the relative corridor settings
    #         # x_min = self.x0 + cfg["corridor_y_min"]
    #         # x_max = self.x0 + cfg["corridor_y_max"]

    #         # W = brne_nav(
    #         #     xtraj, ytraj,
    #         #     1,          # still only the robot’s rollouts
    #         #     tsteps, N,
    #         #     cfg["gp"]["cost_a1"],
    #         #     cfg["gp"]["cost_a2"],
    #         #     cfg["gp"]["cost_a3"],
    #         #     self.ped_sample_scale,
    #         #     x_min,      # absolute left bound in map coords
    #         #     x_max       # absolute right bound
    #         # )


    #         # # 7) First‐step command from NE weights
    #         # w0    = W[0, :N]
    #         # vs    = u_ens[0, :, 0]
    #         # ws    = u_ens[0, :, 1]
    #         # denom = w0.sum()

    #         # # DEBUG: print *before* the branch so we always see the value
    #         # print(f"[DEBUG] w0 sum = {denom},   any NaNs? {np.isnan(denom)}", flush=True)

    #         # if denom > 0 and not np.isnan(denom):
    #         #     v = float((w0 @ vs) / denom)
    #         #     w = float((w0 @ ws) / denom)
    #         #     print(f"[DEBUG] used weighted mean, denom = {denom}", flush=True)
    #         # else:
    #         #     # fallback
    #         #     print(f"[DEBUG] fallback to open‐space cmd, denom = {denom}", flush=True)
    #         #     v = v_os
    #         #     w = yaw_os
                

    #         # # clip & return
    #         # v = np.clip(v, -cfg['max_speed'],   cfg['max_speed'])
    #         # w = np.clip(w, -cfg['max_yaw_rate'], cfg['max_yaw_rate'])
    #         # return v, w




    #     # 1) Sample GPs
    #     robot_trajs, ped_trajs = self.sample_gps(state, ped_list, goal)
    #     self.last_robot_samples = robot_trajs
    #     self.last_ped_samples   = ped_trajs
    #     self.last_ped_trajs     = []

        

    #     # 2) Compute a smooth nominal control toward the goal (as in brne_nav_ros)  citeturn62file15
    #     px, py, th = state[0], state[1], state[2]
    #     # build a perpendicular axis and project to get turning radius
    #     if th > 0.0:
    #         theta_a = th - np.pi/2
    #     else:
    #         theta_a = th + np.pi/2
    #     axis_vec = np.array([np.cos(theta_a), np.sin(theta_a)])
    #     vec2goal = np.array(goal) - np.array([px, py])
    #     dist2goal = np.linalg.norm(vec2goal)
    #     denom = vec2goal @ vec2goal if (vec2goal @ vec2goal) != 0 else 1.0
    #     proj_len = (axis_vec @ vec2goal) / denom * dist2goal
    #     radius = 0.5 * dist2goal / proj_len if proj_len != 0 else np.inf
    #     nominal_vel = cfg["nominal_vel"]
    #     if th > 0.0:
    #         ut = np.array([nominal_vel, -nominal_vel / radius])
    #     else:
    #         ut = np.array([nominal_vel,  nominal_vel / radius])
    #     nominal_cmds = np.tile(ut, (tsteps, 1))
    #     u_ens = get_ulist_essemble(
    #         nominal_cmds,
    #         cfg["max_speed"],
    #         cfg["max_yaw_rate"],
    #         N
    #     )

    #     # 3) Simulate robot rollouts
    #     init_st     = np.tile(state[:3], (N,1)).T    # (3, N)
    #     robot_rolls = traj_sim_essemble(init_st, u_ens, self.dt)
    #     # robot_rolls.shape == (tsteps, 3, N)

    #     # 4) Assemble joint trajectories
    #     num_agents = 1 + len(ped_trajs)
    #     xtraj = np.zeros((num_agents * N, tsteps))
    #     ytraj = np.zeros((num_agents * N, tsteps))

    #     # 4a) robot block
    #     xtraj[0:N, :] = robot_rolls[:, 0, :].T  # (N, tsteps)
    #     ytraj[0:N, :] = robot_rolls[:, 1, :].T

    #     # 4b) pedestrian blocks
    #     for j, ped in enumerate(ped_trajs, start=1):
    #         s, e = j * N, (j + 1) * N
    #         # ped.shape == (N, tsteps, 2)
    #         xtraj[s:e, :] = ped[:, :, 0]
    #         ytraj[s:e, :] = ped[:, :, 1]

    #     # 5) Run the BRNE optimizer
    #     W = brne_nav(
    #         xtraj, ytraj,
    #         num_agents, tsteps, N,
    #         cfg["gp"]["cost_a1"],
    #         cfg["gp"]["cost_a2"],
    #         cfg["gp"]["cost_a3"],
    #         self.ped_sample_scale,
    #         cfg.get("corridor_y_min"),
    #         cfg.get("corridor_y_max"),
    #     )
    #     self.last_W = W

    #     # 6) Extract each pedestrian's NE trajectory
    #     for pi in range(len(ped_trajs)):
    #         s, e = (pi + 1) * N, (pi + 2) * N
    #         rows = xtraj[s:e, :]  # shape (N, tsteps)
    #         cols = ytraj[s:e, :]

    #         w_i_raw = W[0, s:e]   # ideally length N
    #         # ─── GUARD against bad weight length ───
    #         if w_i_raw.shape[0] != N or w_i_raw.sum() == 0.0:
    #             # skip this pedestrian if no valid weights
    #             continue
    #         w_i = w_i_raw

    #         x_ne = (rows * w_i[:, None]).sum(axis=0) / w_i.sum()
    #         y_ne = (cols * w_i[:, None]).sum(axis=0) / w_i.sum()
    #         self.last_ped_trajs.append(np.stack([x_ne, y_ne], axis=1))

    #     # # 7) First‐step robot command from NE weights
    #     # w0 = W[0, :N]
    #     # vs = u_ens[0, :, 0]
    #     # ws = u_ens[0, :, 1]
    #     # v  = float(np.dot(w0, vs) / w0.sum())
    #     # w  = float(np.dot(w0, ws) / w0.sum())

    #     # 7) First‐step robot command from NE weights
    #     w0    = W[0, :N]
    #     vs    = u_ens[0, :, 0]
    #     ws    = u_ens[0, :, 1]
    #     denom = w0.sum()

    #     # ── guard zero or NaN denominator ───────────────────────────
    #     if denom > 0 and not np.isnan(denom):
    #         v = float((w0 @ vs) / denom)
    #         w = float((w0 @ ws) / denom)
    #     else:
    #         # fallback to the smooth nominal command (ut) if sampling failed
    #         # note: 'ut' was defined earlier as the nominal [v,ω] toward the goal
    #         v, w = ut
    #     # ─────────────────────────────────────────────────────────────

    #     # clip & return
    #     v = np.clip(v, -cfg['max_speed'],   cfg['max_speed'])
    #     w = np.clip(w, -cfg['max_yaw_rate'], cfg['max_yaw_rate'])


    #     return v, w

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
