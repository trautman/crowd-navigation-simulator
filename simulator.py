#!/usr/bin/env python3
import argparse, os, yaml, math, random
import numpy as np
import matplotlib.pyplot as plt
import rvo2
import time

from matplotlib.patches import Rectangle, Wedge
from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection

from dwa_controller import DWAController
from brne_controller import BRNEController
from spawner_scheduler import Spawner
from visualization import visualize_dwa, visualize_brne

TIME_STEP = 0.1

def load_config(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def add_walls(sim, walls):
    for w in walls:
        x,y,sx,sy = w['pos_x'], w['pos_y'], w['scale_x'], w['scale_y']
        x0,x1 = x - sx/2, x + sx/2
        y0,y1 = y - sy/2, y + sy/2
        corners = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        for i in range(4):
            sim.addObstacle([corners[i], corners[(i+1)%4]])
    sim.processObstacles()


def random_spawners_from_env(env_cfg, sim_cfg, robot_start=None):
    """
    Overrides the old uniform‐corridor sampling.
    Now we have 4 fixed “blocks”:
      block1: 3<x<7,  -11<y<-8  ↔ goal in block3 (2<x<5, 11<y<14)
      block2:10<x<11,  7<y<9   ↔ goal in block4 (3<x<7,  0<y<5)
      block3: 2<x<5,  11<y<14  ↔ goal in block1 (3<x<7, -11<y<-8)
      block4: 2.5<x<3,   -2.5<y<2.5   ↔ goal in block2 (10<x<11,7<y<9)
    Ignores robot_start since these blocks lie well outside it.
    """

    # # define your four blocks
    # blocks = {
    #   'block1': ((5,  8),   (-11, -8)),
    #   'block2': ((10, 11),   (7,   9)),
    #   'block3': ((4,  5.5),    (11,  14)),
    #   'block4': ((2.5,  3),    (-5,    2.5)),
    # }
    # # mapping start→goal
    # pair = {
    #   'block1':'block3',
    #   'block3':'block1',
    #   'block2':'block4',
    #   'block4':'block2'
    # }
    # pick blocks & pair definitions based on env file
    env_file = env_cfg.get('_env_file', '')
    if 'perks' in env_file:
        # ── PERKS environment spawner blocks ───────────────────────
        blocks = {
          'pblock1': ((0, 5), (-12, -7)),
          'pblock2': ((0, 5), (8, 10)),
          # … fill in your perks‐specific coordinate ranges …
        }
        pair = {
          'pblock1': 'pblock2',
          'pblock2': 'pblock1',
          # … etc. …
        }
    else:
        # ── BOARDWALK environment spawner blocks ────────────────
        blocks = {
          'block1': ((5,   8),    (-11,  -8)),
          'block2': ((10, 11),    (7,     9)),
          'block3': ((4,   5.5),  (11,   14)),
          'block4': ((2.5, 3),    (-5,   2.5)),
        }
        pair = {
          'block1': 'block3',
          'block3': 'block1',
          'block2': 'block4',
          'block4': 'block2',
        }

    rate_min = sim_cfg.get('spawn_rate_min', 0.1)
    rate_max = sim_cfg.get('spawn_rate_max', 1.0)
    duration = sim_cfg['simulation']['duration']

    spawners = []
    for _ in sim_cfg['ped_spawners']:
        # pick a random start block
        start_blk = random.choice(list(pair.keys()))
        x_rng, y_rng = blocks[start_blk]
        spawn = np.array([
            random.uniform(*x_rng),
            random.uniform(*y_rng)
        ])

        # pick the corresponding goal block
        goal_blk = pair[start_blk]
        gx_rng, gy_rng = blocks[goal_blk]
        goal = np.array([
            random.uniform(*gx_rng),
            random.uniform(*gy_rng)
        ])

        # spawn frequency schedule
        freq  = random.uniform(rate_min, rate_max)
        sched = [{'end': duration, 'period': freq}]

        spawners.append(Spawner(spawn, goal, sched))

    return spawners


# def run_sim(env_conf, sim_conf, gui=False):
def run_sim(env_conf, sim_conf, gui=False, output_base=None):
    print(f"GUI = {'ON' if gui else 'OFF'}")
    env_cfg = load_config(env_conf)
    sim_cfg = load_config(sim_conf)
    env_cfg['_env_file'] = os.path.basename(env_conf).lower()


    # — if loading the "perks" environment, rotate it 90° clockwise —
    if os.path.basename(env_conf).lower() == 'perks.yaml':
        def _rotate90cw(item):
            # rotate point (x,y) → (y, -x)
            x, y = item['pos_x'], item['pos_y']
            item['pos_x'], item['pos_y'] = y, -x
            # and spin the object itself by +90°
            if 'rot_z' in item:
                item['rot_z'] = (item.get('rot_z', 0.0) + 90.0) % 360.0
            return item

        # apply to floors and walls lists in place
        env_cfg['floors'] = [_rotate90cw(f) for f in env_cfg.get('floors', [])]
        env_cfg['walls']  = [_rotate90cw(w) for w in env_cfg.get('walls', [])]

    # ── Visualization settings ───────────────────────────────
    viz_cfg      = sim_cfg.get('visualization', {})
    report_interval = float(
        sim_cfg.get('simulation', {}).get('report_interval', 10.0))
    ENABLE_DWA_VIZ  = viz_cfg.get('dwa', False)
    ENABLE_BRNE_VIZ = viz_cfg.get('brne', False)
    VIZ_PAUSE     = viz_cfg.get('pause_between', TIME_STEP)


    floors = env_cfg.get('floors',[])
    walls  = env_cfg.get('walls', [])

    S  = sim_cfg['simulation']
    duration    = S['duration']
    robot_delay = S['robot_delay']
    goal_tol    = S['goal_tolerance']
    FOV_DEG     = S['fov_deg']
    FOV_R       = S['fov_range']
    close_th    = S['close_stop_threshold']
    agent_speed = S.get('agent_speed',1.0)
    n_trials    = S.get('n_trials',1)

    R     = sim_cfg['robot']
    algo  = R['algorithm'].upper()
    print(f"Robot navigation algorithm = {algo}")
    start = np.array(R['start'])
    goal  = np.array(R['goal'])


    base = output_base or '.'

    out_dir       = os.path.join(base, 'density')
    safety_dir    = os.path.join(base, 'safety_distances')
    trans_vel_dir = os.path.join(base, 'translational_velocity')
    path_len_dir  = os.path.join(base, 'path_length')
    time_dir      = os.path.join(base, 'travel_time')
    stop_time_dir = os.path.join(base, 'time_not_moving')
    # out_dir            = 'density'
    # safety_dir         = 'safety_distances'
    # trans_vel_dir      = 'translational_velocity'
    # path_len_dir       = 'path_length'
    # time_dir           = 'travel_time'
    # stop_time_dir      = 'time_not_moving'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(safety_dir, exist_ok=True)
    os.makedirs(trans_vel_dir, exist_ok=True)
    os.makedirs(path_len_dir, exist_ok=True)
    os.makedirs(time_dir, exist_ok=True)
    os.makedirs(stop_time_dir, exist_ok=True)




    for trial in range(1, n_trials+1):
        print(f"\n=== Trial {trial}/{n_trials} ===")
        # record real‐wall clock start for this trial
        trial_wall_start = time.perf_counter()

        next_report = report_interval

        # robot controller
        if algo=='BRNE':
            rcfg      = sim_cfg['brne']
            robot_ctl = BRNEController(rcfg, TIME_STEP)
            max_spd   = rcfg['max_speed']
            max_yaw   = rcfg['max_yaw_rate']
        else:
            rcfg      = sim_cfg['dwa']
            robot_ctl = DWAController(rcfg, TIME_STEP)
            max_spd   = rcfg['max_speed']
            max_yaw   = rcfg['max_yaw_rate']

        # spawners
        if sim_cfg.get('spawn_mode','fixed').lower()=='random':
            ped_spawners = random_spawners_from_env(env_cfg, sim_cfg, start)
        else:
            ped_spawners=[]
            for sp in sim_cfg['ped_spawners']:
                pos = np.array(sp['pos'])
                goal_pt = np.array(sp['goal'])
                freq = sp.get('freq', sp.get('spawn_frequency',1.0))
                sched= [{'end':duration,'period':freq}]
                ped_spawners.append(Spawner(pos,goal_pt,sched))

        ped_agents=[]

        # ORCA sim
        O = sim_cfg['orca']
        sim = rvo2.PyRVOSimulator(
            TIME_STEP,
            O['neighbor_dist'],O['max_neighbors'],
            O['time_horizon'],O['time_horizon_obst'],
            O['agent_radius'], agent_speed
        )
        add_walls(sim, walls)
        rid = sim.addAgent(tuple(start))
        sim.setAgentMaxSpeed(rid, max_spd)

        # metrics storage
        # density=[]
        # safety_distances = []
                # metrics storage
        density              = []
        safety_distances     = []
        translational_vels   = []
        dist_traveled        = []
        time_elapsed_list    = []
        time_not_moving_list = []
        prev_pos = sim.getAgentPosition(rid)

        fov_area = (FOV_DEG/360.0)*math.pi*(FOV_R**2)
        t=0.0
        achieved=False

        # initial robot state
        th0 = math.atan2(goal[1]-start[1], goal[0]-start[0])
        rstate = [start[0], start[1], th0, 0.0, 0.0]

        # GUI setup
        if gui:
            plt.ion()
            fig,ax = plt.subplots(figsize=(8,8))
            ax.set_aspect('equal')
            # floors & walls
            for f in floors:
                fx,fy = f['pos_x'], f['pos_y']
                fw,fh = f['scale_x'],f['scale_y']
                ang   = f.get('rot_z',0.0)
                rect  = Rectangle((-fw/2,-fh/2),fw,fh,color='lightgray',zorder=0)
                rect.set_transform(Affine2D().rotate_deg_around(0,0,-ang)
                                   .translate(fx,fy)+ax.transData)
                ax.add_patch(rect)
            for w in walls:
                ax.add_patch(Rectangle(
                    (w['pos_x']-w['scale_x']/2, w['pos_y']-w['scale_y']/2),
                    w['scale_x'], w['scale_y'], color='saddlebrown', zorder=1
                ))
            # spawners & goal/start
            for sp in ped_spawners:
                ax.scatter(*sp.pos, c='green', marker='x', s=80, zorder=2)
                ax.scatter(*sp.goal,c='red',   marker='*', s=80, zorder=2)
            ax.scatter(*start,c='magenta',marker='s', s=80, label='Start',zorder=3)
            ax.scatter(*goal, c='magenta',marker='D', s=80, label='Goal', zorder=3)
            ax.legend(loc='upper left')

            scatter = ax.scatter([],[],s=50,zorder=4)
            robot_patch = Rectangle((-0.3,-0.3),0.6,0.6,
                                    facecolor='magenta',edgecolor='black',zorder=5)
            ax.add_patch(robot_patch)
            wedge = Wedge((0,0),FOV_R,0,0,facecolor='yellow',alpha=0.3,zorder=2)
            ax.add_patch(wedge)
            plt.draw(); plt.pause(0.001)
            frame_count = 0
            
            # ── Pre-allocate GP & NE collections ───────────────────────────────
            # (LineCollection already imported at top of file)
            # gp_collection = LineCollection(
            #     [], linewidths=1, alpha=0.3, colors='red', zorder=2
            # )
            # ax.add_collection(gp_collection)
            
            ne_collection = LineCollection(
                [], linewidths=2, colors='black', zorder=3
            )
            ax.add_collection(ne_collection)
            
            # ── Pre-allocate DWA visualization ────────────────────────────────
            dwa_cands = LineCollection(
                [], linewidths=0.5, alpha=0.3, colors='blue', zorder=2
            )
            ax.add_collection(dwa_cands)
            
            dwa_best_line, = ax.plot(
                [], [], linewidth=2, color='darkblue', label='DWA best', zorder=3
            )
            ax.legend(loc='upper left')



            gp_lines = []
            ne_lines = []
            frame_count = 0

        frame_count = 0
        # run simulation loop
        while t < duration:
            frame_count += 1
            dt = TIME_STEP
            
            for sp in ped_spawners:
                p = sp.current_period(t)
                if t >= sp.last_spawn + p:
                    aid = sim.addAgent(tuple(sp.pos))
                    ped_agents.append({'id':aid,'goal':sp.goal})
                    sp.last_spawn = t

            # FOV
            vis=[]
            rx,ry,th = rstate[:3]
            for a in ped_agents:
                px,py = sim.getAgentPosition(a['id'])
                dx,dy = px-rx, py-ry
                d = math.hypot(dx,dy)
                ang = math.degrees(math.atan2(dy,dx)-th)
                rel = (ang+180)%360 - 180
                if d<=FOV_R and abs(rel)<=FOV_DEG/2:
                    vis.append({'dist':d,'goal':a['goal']})

            # robot control
            prev = list(rstate)
            if t>=robot_delay:
                # # density
                # cnt = sum(1 for p in vis if p['dist']<=FOV_R)
                # density.append(cnt / fov_area)
                # density
                cnt = sum(1 for p in vis if p['dist']<=FOV_R)
                density.append(cnt / fov_area)
                # safety: minimum distance to any ped in FOV (or FOV_R if none)
                if vis:
                    min_d = min(p['dist'] for p in vis)
                else:
                    min_d = FOV_R
                safety_distances.append(min_d)


                           # record new metrics
                translational_vels.append(v)
                time_elapsed_list.append(dt)
                cur_pos = sim.getAgentPosition(rid)
                dx, dy = cur_pos[0] - prev_pos[0], cur_pos[1] - prev_pos[1]
                dist = math.hypot(dx, dy)
                dist_traveled.append(dist)
                prev_pos = cur_pos
                if abs(v) < 1e-3 and abs(w) < 1e-3:
                    time_not_moving_list.append(dt)
                else:
                    time_not_moving_list.append(0.0)


                if any(p['dist']<=close_th for p in vis):
                    v,w = 0.0,0.0
                else:
                    ped_list_ctrl = [
                        {'id': a['id'],
                         'pos': sim.getAgentPosition(a['id']),
                         'goal': a['goal']}
                        for a in ped_agents
                    ] 

                    # 2) FOV filter (shared)
                    rx, ry, rth = rstate[:3]
                    ped_list_fov = []
                    for p in ped_list_ctrl:
                        dx, dy = p['pos'][0] - rx, p['pos'][1] - ry
                        dist    = math.hypot(dx, dy)
                        ang     = math.degrees(math.atan2(dy, dx) - rth)
                        rel     = (ang + 180) % 360 - 180
                        if dist <= FOV_R and abs(rel) <= FOV_DEG/2:
                            ped_list_fov.append(p)

                    if algo == 'DWA':
                        obs = [p['pos'] for p in ped_list_fov]
                        robot_ctl.cfg['current_v'] = rstate[3]
                        v, w = robot_ctl.control(rstate, goal, obs)
                    else:
                            # note: BRNEController expects dicts with 'pos' and 'goal'
                        t0 = time.perf_counter()
                        v, w = robot_ctl.control(rstate, goal, ped_list_fov)
                        t1 = time.perf_counter()
                        # you can log (t1-t0) if you like

                        # clip & apply
                        v = np.clip(v, -max_spd, max_spd)
                        w = np.clip(w, -max_yaw, max_yaw)
                        sim.setAgentPrefVelocity(
                            rid,
                            (v * math.cos(rstate[2]), v * math.sin(rstate[2]))
                        )  

                v = np.clip(v,-max_spd,max_spd)
                w = np.clip(w,-max_yaw,max_yaw)
                sim.setAgentPrefVelocity(rid, (v*math.cos(th), v*math.sin(th)))
            else:
                v,w = 0.0, 0.0
                sim.setAgentPrefVelocity(rid,(0.0,0.0))

            # ped pref-vel
            for a in ped_agents:
                p = np.array(sim.getAgentPosition(a['id']))
                vec = a['goal'] - p; dd = np.linalg.norm(vec)
                pv = tuple((vec/dd)*agent_speed if dd>1e-3 else (0.0,0.0))
                sim.setAgentPrefVelocity(a['id'],pv)

            # step
            sim.doStep()

            # override robot state
            th = rstate[2]
            rstate[0] += v*math.cos(th)*TIME_STEP
            rstate[1] += v*math.sin(th)*TIME_STEP
            rstate[2] += w*TIME_STEP
            rstate[2] = (rstate[2]+math.pi)%(2*math.pi)-math.pi
            rstate[3], rstate[4] = v,w
            sim.setAgentPosition(rid,(rstate[0],rstate[1]))
            sim.setAgentVelocity(rid,(v*math.cos(rstate[2]),v*math.sin(rstate[2])))

            # GUI update
            if gui:

                # build positions + per-agent colors based on FOV
                pts = [sim.getAgentPosition(a['id']) for a in ped_agents]
                colors = []
                rx, ry, rth = rstate[:3]
                for (x, y), a in zip(pts, ped_agents):
                    dx, dy = x - rx, y - ry
                    dist    = math.hypot(dx, dy)
                    rel_ang = (math.degrees(math.atan2(dy, dx) - rth) + 180) % 360 - 180
                    # red if in FOV, else gray
                    if dist <= FOV_R and abs(rel_ang) <= FOV_DEG/2:
                        colors.append('red')
                    else:
                        colors.append('gray')
                if pts:
                    scatter.set_offsets(pts)
                    scatter.set_color(colors)

                # update robot orientation & position
                tr = Affine2D().rotate_deg_around(
                    0, 0, math.degrees(rstate[2])
                ).translate(rstate[0], rstate[1])
                robot_patch.set_transform(tr + ax.transData)

                # always update wedge center
                wedge.set_center((rstate[0], rstate[1]))

                # compute raw angles
                th_deg = math.degrees(rstate[2])
                theta1 = th_deg - FOV_DEG/2    # <<< unchanged, just compute
                theta2 = th_deg + FOV_DEG/2    # <<< unchanged, just compute

                # only update the wedge angles (no other code here!) if valid
                if not math.isnan(theta1) and not math.isnan(theta2):
                    wedge.theta1 = theta1      # <<< guarded
                    wedge.theta2 = theta2      # <<< guarded

                plt.pause(TIME_STEP)



            # prune
            ped_agents = [a for a in ped_agents
                if np.linalg.norm(np.array(sim.getAgentPosition(a['id']))-a['goal'])>0.5]

            # check goal
            if not achieved and t>=TIME_STEP and np.linalg.norm(rstate[:2]-goal)<goal_tol:
                achieved = True
                break

            t += TIME_STEP

            # ── report progress every report_interval s ──────────────────────────────
            if t >= next_report:
                # compute wall‐clock elapsed
                wall_elapsed = time.perf_counter() - trial_wall_start
                print(
                    f"[Trial {trial}] sim {t:.1f}/{duration:.1f}s    "
                    f"real {wall_elapsed:.1f}s"
                )
                next_report += report_interval

        # write density metrics
        outpath = os.path.join(out_dir, f"density_trial_{trial}.txt")
        with open(outpath, 'w') as f:
            for ρ in density:
                f.write(f"{ρ:.6f}\n")
        print(f"Trial {trial} {'OK' if achieved else 'FAIL'} → {outpath}")

        # write safety‐distance metrics
        safe_out = os.path.join(safety_dir, f"safety_distances_trial_{trial}.txt")
        with open(safe_out, 'w') as f:
            for d in safety_distances:
                f.write(f"{d:.6f}\n")
        print(f"Safety distances saved to {safe_out}")

                # write translational velocities
        tv_out = os.path.join(trans_vel_dir, f"translational_velocity_trial_{trial}.txt")
        with open(tv_out, 'w') as f:
            for tv in translational_vels:
                f.write(f"{tv:.6f}\n")

        # write path lengths
        pl_out = os.path.join(path_len_dir, f"path_length_trial_{trial}.txt")
        with open(pl_out, 'w') as f:
            for d in dist_traveled:
                f.write(f"{d:.6f}\n")

        # write elapsed times
        te_out = os.path.join(time_dir, f"travel_time_trial_{trial}.txt")
        with open(te_out, 'w') as f:
            for te in time_elapsed_list:
                f.write(f"{te:.6f}\n")

        # write time-not-moving
        tn_out = os.path.join(stop_time_dir, f"time_not_moving_trial_{trial}.txt")
        with open(tn_out, 'w') as f:
            for tm in time_not_moving_list:
                f.write(f"{tm:.6f}\n")

        if gui:
            plt.clf()

    print("\nAll done.")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env-config', default='boardwalk.yaml')
    p.add_argument('--sim-config', default='simulator_config.yaml')
    p.add_argument('-g','--gui', action='store_true',
                   help='Live-plot each trial')
    # args = p.parse_args()
    # run_sim(args.env_config, args.sim_config, gui=args.gui)
    p.add_argument('--write_data_to', default=None,
                   help='Optional base directory for saving metric outputs')
    args = p.parse_args()
    run_sim(args.env_config, args.sim_config, gui=args.gui, output_base=args.write_data_to)

