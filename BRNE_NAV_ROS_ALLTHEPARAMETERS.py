#############################################
import numpy as np
from . import brne as brne
from .traj_tracker import TrajTracker

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_2d_msgs.msg import Twist2D
from geometry_msgs.msg import PoseStamped, PoseArray, Pose2D, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from social_nav_msgs.msg import PedestriansWithCovariance, Pedestrian
from social_nav_msgs.msg import PedestrianWithCovariance, Pedestrians
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from visualization_msgs.msg import Marker, MarkerArray
from benchmark_utils import BenchmarkPub
from std_msgs.msg import Int16, String
from action_msgs.msg import GoalStatus
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from rclpy.time import Time
from rclpy.duration import Duration
from simple_actions import SimpleActionServer
from nav2_msgs.action import NavigateToPose
import time
import yaml


def pose2d_transform(msg):
    roll, pitch, yaw = euler_from_quaternion(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    return np.array([msg.position.x, msg.position.y, yaw])


# THIS IS THE NAVIGATION CLASS !!!
class BrneNavRos(Node):
    def __init__(self):
        super().__init__('brne_nav_ros')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.ped_info_pub = self.create_publisher(PoseArray, '/brne_peds', 1)
        self.opt_traj_pub = self.create_publisher(Path, '/optimal_path', 1)
        self.brne_traj_pub = self.create_publisher(Path, '/brne_path', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/brne_markers', 1)
        self.num_peds_pub = self.create_publisher(Int16, '/brne/n_pedestrians', 1)
        self.benchmark_pub = BenchmarkPub(self, '/brne/compute_time')
        self.result_pub = self.create_publisher(GoalStatus, '/navigation_result', 1)
        self.log_pub = self.create_publisher(String, '/brne/log', 1)

        latching_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL
        )
        self.params_pub = self.create_publisher(String, '/brne/params', qos_profile=latching_qos)

        ####################################################################################
        # ALL PARAMETERS ARE DEFINED HERE!!!
        ####################################################################################
        self.declare_parameter('maximum_agents', 5)  # maximum number of agents BRNE will consider (including the robot)
        self.declare_parameter('num_samples', 196)  # number of samples assigned to each agent
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('plan_steps', 25)  # time steps of the planning horizon
        self.declare_parameter('max_lin_vel', 0.6)  # 0.6,0.8 maximum linear velocity allowed on the robot
        self.declare_parameter('max_ang_vel', 1.0)  # 1.0,1.2 maximum angular velocity allowed on the robot
        self.declare_parameter('nominal_vel', 0.4)  #0.4,0.5 nomimal (linear) velocity when plannig the initial trajectory
        self.declare_parameter('kernel_a1', 0.2)  # control the "straightness" of trajectory samples. Larger the value is, less straight the trajectory sample will be.
        self.declare_parameter('kernel_a2', 0.2)  # control the "width/spreadness" of trajectory samples. Larger the value is, more spread the trajectory samples are.
        self.declare_parameter('cost_a1', 4.0)  # control the safety zone, smaller the value is, more conversative the robot will be.
        self.declare_parameter('cost_a2', 1.0)  # control the safety zone, larger the value is, more conservative the robot will be.
        self.declare_parameter('cost_a3', 80.0)  #  control the safety penalty weight, larger the value is, more conservative the robot will be.
        self.declare_parameter('ped_sample_scale', 0.1)  # pedestrian's willingness for cooperation, default value is 1.0, the smaller it is, the less the robot would expect the pedestrians to make space for it
        self.declare_parameter('ad', -5.0)  # "aggressiveness" of the optimal controller, the more negative ad is, the more aggressive the robot will be on chasing the way points (might leads to jerky or aggressive behavior)
        self.declare_parameter('R_lin', 1.0)  # control penalty weight on linear velocity, the larger it is, the smoother the linear motion will be, but way point tracking accuracy might be compromised
        self.declare_parameter('R_ang', 2.0)  # control penalty weight on angular velocity, the larger it is, the smoother the rotation motion will be, but way point tracking accuracy might be compromised
        self.declare_parameter('replan_freq', 10.0)  # unit: Hz
        self.declare_parameter('people_timeout', 5.0)  # unit: seconds
        self.declare_parameter('corridor_y_min', -1.0)  # lower bound of y coordinate (one side of corridor)
        self.declare_parameter('corridor_y_max', 1.0)  # upper bound of y coordinate (the other side of corridor)
        self.declare_parameter('frame2frame_enabled', False)   # Whether to use frame to frame for velocity calculation, or use incoming data
        self.declare_parameter('staircase_truncation', False)  # saturate F2F velocity in a staircase manner
        self.declare_parameter('people_timeout_off', True)
        self.declare_parameter('close_stop_threshold', 0.5)  # threshold for safety mask, leading to estop
        self.declare_parameter('open_space_velocity', 0.6)  # nominal velocity when the robot is in open space
        self.declare_parameter('brne_activate_threshold', 3.5)  # distance threshold from a pedestrian to enable BRNE
        self.declare_parameter('point_cloud_dist_threshold', 0.5)  # safety distance threshold to stereo point cloud
        self.declare_parameter('sensor_patience', 2.0)  # Amount of seconds to allow navigation even when there is no recent pedestrian data
        ####################################################################################


        self.num_agents = self.get_parameter('maximum_agents').value
        self.num_samples = self.get_parameter('num_samples').value
        self.dt = self.get_parameter('dt').value
        self.plan_steps = self.get_parameter('plan_steps').value
        self.max_lin_vel = self.get_parameter('max_lin_vel').value
        self.max_ang_vel = self.get_parameter('max_ang_vel').value
        self.nominal_vel = self.get_parameter('nominal_vel').value
        self.kernel_a1 = self.get_parameter('kernel_a1').value
        self.kernel_a2 = self.get_parameter('kernel_a2').value
        self.cost_a1 = self.get_parameter('cost_a1').value
        self.cost_a2 = self.get_parameter('cost_a2').value
        self.cost_a3 = self.get_parameter('cost_a3').value
        self.ped_sample_scale = self.get_parameter('ped_sample_scale').value
        self.replan_freq = self.get_parameter('replan_freq').value
        self.people_timeout = Duration(seconds=self.get_parameter('people_timeout').value)
        self.corridor_y_min = self.get_parameter('corridor_y_min').value
        self.corridor_y_max = self.get_parameter('corridor_y_max').value
        self.frame2frame_enabled = self.get_parameter('frame2frame_enabled').value
        self.staircase_truncation = self.get_parameter('staircase_truncation').value
        self.people_timeout_off = self.get_parameter('people_timeout_off').value
        self.close_stop_threshold = self.get_parameter('close_stop_threshold').value
        self.open_space_velocity = self.get_parameter('open_space_velocity').value
        self.brne_activate_threshold = self.get_parameter('brne_activate_threshold').value
        self.point_cloud_dist_threshold = self.get_parameter('point_cloud_dist_threshold').value
        self.sensor_patience = rclpy.duration.Duration(seconds=self.get_parameter('sensor_patience').value)

        self.params_msg = String()
        param_dict = {}
        for param in self._parameters:
            param_dict[param] = self.get_parameter(param).value
        self.params_msg.data = yaml.safe_dump(param_dict)

        self.num_peds = 0  # number of current pedestrians that BRNE chooses to interact with
        self.last_ped_stamp = None

        parallel_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()

        sensor_qos = rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value

        self.ped_sub = self.create_subscription(PedestriansWithCovariance,
                                                '/top/zed/obj_det/pedestrians', self.ped_cb, 1,
                                                callback_group=parallel_cb_group)
        self.map_sub = self.create_subscription(OccupancyGrid, '/static_map', self.map_cb, latching_qos,
                                                callback_group=parallel_cb_group)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 1,
                                                 callback_group=parallel_cb_group)
        self.odom_sub2 = self.create_subscription(Odometry, '/true_odom', self.odom_cb, 1,
                                                  callback_group=parallel_cb_group)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_cb, 1,
                                                 callback_group=parallel_cb_group)
        # self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb,
        #                                          qos_profile=sensor_qos,
        #                                          callback_group=parallel_cb_group)
        self.pc_sub = self.create_subscription(PointCloud2, '/top/zed/point_cloud/clean_cloud', self.pc_cb,
                                               qos_profile=sensor_qos,
                                               callback_group=parallel_cb_group)
        self.scan_points = np.ones((10, 2)) * 1000.0

        # this callback function publish control command at fixed frequency
        self.ped_msg_buffer = {}
        self.timer = self.create_timer(self.dt, self.timer_cb,
                                       callback_group=parallel_cb_group)
        self.brne_timer = self.create_timer(1.0/self.replan_freq, self.brne_cb,
                                            callback_group=parallel_cb_group)

        self.robot_pose = np.zeros(3)  # the robot's initial pose
        self.robot_goal = None   # the robot's goal
        self.robot_traj = []

        self.cmd_tracker = TrajTracker(dt=self.dt, max_lin_vel=self.max_lin_vel, max_ang_vel=self.max_ang_vel)   # the class that converts way points to control commands
        self.cmds = np.zeros((self.plan_steps, 2))   # the control command buffer
        self.cmd_counter = 0   # counter that goes with the control command buffer

        # the trajectory from the optimal control commands
        self.cmds_traj = np.tile(self.robot_pose, (self.plan_steps, 1))

        self.x_opt_trajs = np.zeros((self.num_agents, self.plan_steps))  # optimal trajectories from BRNE
        self.y_opt_trajs = np.zeros((self.num_agents, self.plan_steps))

        # initialize the BRNE covariance matrix here
        tlist = np.arange(self.plan_steps) * self.dt
        train_ts = np.array([tlist[0]])
        train_noise = np.array([1e-04])
        test_ts = tlist
        self.cov_Lmat, self.cov_mat = brne.get_Lmat_nb(train_ts, test_ts, train_noise, self.kernel_a1, self.kernel_a2)

        self.cmd_log = []

        ### F2F velocity estimation
        self.curr_ped_array = np.array([])  # np.array([[]])
        self.prev_ped_array = np.array([])  # np.array([[]])

        self.close_stop_flag = False

        self.from_action = False
        self.action_server = SimpleActionServer(self, NavigateToPose, '/navigate_to_pose', self.navigate_action)

    def scan_cb(self, msg:LaserScan):
        self.log('scan_cb')
        num_ranges = len(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, num_ranges)
        ranges = np.array(msg.ranges)
        ranges = np.clip(ranges, a_min=-1000.0, a_max=50.0)
        ranges = np.nan_to_num(ranges, nan=1000.0)
        points_x = np.array(ranges) * np.cos(angles)
        points_y = np.array(ranges) * np.sin(angles)
        original_points = np.array([
            points_x, points_y, np.ones(num_ranges)
        ])

        x, y, th = self.robot_pose.copy()
        pose_se2 = np.array([
            [np.cos(th), -np.sin(th), x],
            [np.sin(th),  np.cos(th), y],
            [0.0, 0.0, 1.0]
        ])
        transformed_points = pose_se2 @ original_points

        self.scan_points = (transformed_points.T)[:,:2]
        # for i, pt in enumerate(self.scan_points):
        #     self.log(f'point[{i}]: {pt[0]}, {pt[1]}')

    def pc_cb(self, msg:PointCloud2):
        # self.scan_points = list(read_points(msg, ['x', 'y'], skip_nans=True))
        msg_points = list(read_points(msg, ['x', 'y'], skip_nans=True))
        num_points = len(msg_points)
        if num_points < 2:  # if not enough points, create some far away fake points to make sure there is data
            points = np.array([
                [100.0, 100.0],
                [101.0, 101.0],
            ])
            self.scan_points = points.copy()
        else:
            points = np.zeros((num_points, 2))
            for i in range(num_points):
                points[i][0] = msg_points[i][0]
                points[i][1] = msg_points[i][1]
            self.scan_points = points.copy()

    def brne_cb(self):
        # if self.robot_goal is None:
        #     return
        self.benchmark_pub.tick('brne_cb')

        xmean_list = np.zeros((self.num_agents, self.plan_steps))
        xmean_list = np.zeros((self.num_agents, self.plan_steps))

        ped_info_list = []
        dists2peds = []

        now = self.get_clock().now()

        if self.people_timeout_off == False:
            for ped_ident, (ped, stamp) in list(self.ped_msg_buffer.items()):
                if now - stamp > self.people_timeout:
                    del self.ped_msg_buffer[ped_ident]

        # we go through each perceived pedestrian and save the information
        for ped, stamp in self.ped_msg_buffer.values():
            dist2ped = np.sqrt((self.robot_pose[0]-ped.pose.x)**2 + (self.robot_pose[1]-ped.pose.y)**2)
            if dist2ped < self.brne_activate_threshold:  # only consider pedestrians within the activate threshold
                ped_info = np.array([
                    ped.pose.x, ped.pose.y, ped.velocity.x, ped.velocity.y
                ])
                ped_info_list.append(ped_info)

                dists2peds.append(dist2ped)

        ped_info_list = np.array(ped_info_list)
        self.num_peds = len(ped_info_list)

        dists2peds = np.array(dists2peds)

        # compute how many pedestrians we are actually interacting with
        num_agents = np.minimum(self.num_peds+1, self.num_agents)

        # Publish num agents
        num_peds_msg = Int16()
        num_peds_msg.data = int(num_agents-1)
        self.num_peds_pub.publish(num_peds_msg)

        self.log(f'total # pedestrians: {self.num_peds}.')
        self.log(f'brne # agents: {num_agents}.')

        # self.num_peds = 0
        if num_agents > 1:
            ped_indices = np.argsort(dists2peds)[:num_agents-1]  # we only pick the N closest pedestrian to interact with
            robot_state = self.robot_pose.copy()

            x_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)
            y_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples, self.plan_steps, self.cov_Lmat)

            # ctrl space configuration here
            xtraj_samples = np.zeros((
                num_agents * self.num_samples, self.plan_steps
            ))
            ytraj_samples = np.zeros((
                num_agents * self.num_samples, self.plan_steps
            ))

            closest_dist2ped = 100.0
            closest_ped_pos = np.zeros(2) + 100.0
            for i, ped_id in enumerate(ped_indices):
                ped_pos = ped_info_list[ped_id][:2]
                ped_vel = ped_info_list[ped_id][2:]
                speed_factor = np.linalg.norm(ped_vel)
                ped_xmean = ped_pos[0] + np.arange(self.plan_steps) * self.dt * ped_vel[0]
                ped_ymean = ped_pos[1] + np.arange(self.plan_steps) * self.dt * ped_vel[1]

                xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    x_pts[i*self.num_samples : (i+1)*self.num_samples] * speed_factor + ped_xmean
                ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    y_pts[i*self.num_samples : (i+1)*self.num_samples] * speed_factor + ped_ymean

                dist2ped = np.linalg.norm([
                    robot_state[:2] - ped_pos[:2]
                ])
                if dist2ped < closest_dist2ped:
                    closest_dist2ped = dist2ped
                    closest_ped_pos = ped_pos.copy()

            st = robot_state.copy()

            if self.robot_goal is None:
                goal = np.array([6.0, 0.0])
            else:
                goal = self.robot_goal[:2]

            if st[2] > 0.0:
                theta_a = st[2] - np.pi/2
            else:
                theta_a = st[2] + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])
            vec2goal = goal - st[:2]
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            radius = 0.5 * dist2goal / proj_len

            if st[2] > 0.0:
                ut = np.array([self.nominal_vel, -self.nominal_vel/radius])
            else:
                ut = np.array([self.nominal_vel, self.nominal_vel/radius])
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1))

            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, self.max_lin_vel, self.max_ang_vel, self.num_samples
            )
            traj_essemble = brne.traj_sim_essemble(
                np.tile(robot_state, reps=(self.num_samples,1)).T,
                ulist_essemble,
                self.dt
            )
            xtraj_samples[0:self.num_samples] = traj_essemble[:,0,:].T
            ytraj_samples[0:self.num_samples] = traj_essemble[:,1,:].T

            # generate sample weight mask for the closest pedestrian
            robot_xtrajs = traj_essemble[:,0,:].T
            robot_ytrajs = traj_essemble[:,1,:].T
            robot_samples2ped = (robot_xtrajs - closest_ped_pos[0])**2 + (robot_ytrajs - closest_ped_pos[1])**2
            robot_samples2ped = np.min(np.sqrt(robot_samples2ped), axis=1)
            safety_mask = (robot_samples2ped > self.close_stop_threshold).astype(float)
            safety_samples_percent = safety_mask.mean() * 100
            self.log('percent of safe samples: {:.2f}%'.format(safety_samples_percent))
            self.log('dist 2 ped: {:.2f} m'.format(closest_dist2ped))

            self.close_stop_flag = False
            if np.max(safety_mask) == 0.0:
                safety_mask = np.ones_like(safety_mask)
                self.close_stop_flag = True
            # self.log('safety mask: {}'.format(safety_mask))

            # BRNE OPTIMIZATION HERE !!!
            weights = brne.brne_nav(
                xtraj_samples, ytraj_samples,
                num_agents, self.plan_steps, self.num_samples,
                self.cost_a1, self.cost_a2, self.cost_a3, self.ped_sample_scale,
                self.corridor_y_min, self.corridor_y_max
            )

            # check collision with the point cloud
            point_cloud_safety_idx, point_cloud_min_dist = brne.point_cloud_safety_idx(
                traj_essemble,
                self.scan_points.copy(),
                self.point_cloud_dist_threshold
            )
            point_cloud_safety_idx = np.array(point_cloud_safety_idx)
            if len(point_cloud_safety_idx.shape) == 2:
                point_cloud_safety_idx = point_cloud_safety_idx[0]
            point_cloud_safety_mask = np.zeros(self.num_samples)
            point_cloud_safety_mask[point_cloud_safety_idx] = 1.0

            # apply safety mask
            full_safety_mask = safety_mask * point_cloud_safety_mask
            safe_sample_percentage = 100 * np.sum(full_safety_mask) / self.num_samples
            self.log(f'safe sample percentage: {safe_sample_percentage}%')
            weights[0] *= safety_mask * point_cloud_safety_mask
            if np.mean(weights[0]) == 0:
                weights[0] = np.zeros(self.num_samples)
            else:
                weights[0] /= np.mean(weights[0])



            # self.get_logger().info('point_cloud_min_dist: ')
            # for i in range(self.num_samples):
            #     self.log(f'[{i}]: {point_cloud_min_dist[0]}, {weights[0][i]}')

            # generate optimal ctrl cmds and update buffer
            opt_cmds_1 = np.mean(ulist_essemble[:,:,0] * weights[0], axis=1)
            opt_cmds_2 = np.mean(ulist_essemble[:,:,1] * weights[0], axis=1)
            self.cmds = np.array([opt_cmds_1, opt_cmds_2]).T
            self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)

            ped_trajs_x = np.zeros((num_agents-1, self.plan_steps))
            ped_trajs_y = np.zeros((num_agents-1, self.plan_steps))
            for i in range(num_agents-1):
                ped_trajs_x[i] = \
                    np.mean(xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
                ped_trajs_y[i] = \
                    np.mean(ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)

            self.publish_trajectory(self.opt_traj_pub, self.cmds_traj[:,0], self.cmds_traj[:,1])
            self.publish_markers(ped_trajs_x, ped_trajs_y)

            if self.robot_goal is None or self.close_stop_flag == True:
                self.cmds = np.zeros((self.plan_steps, 2))
                self.cmds_traj = np.tile(robot_state, reps=(self.plan_steps,1))

            # for smoothness, we allow the robot to execute the first 5 time steps from the buffer
            if self.cmd_counter > 0:
                self.cmd_counter = 0

        else:  # if no pedestrian around, go straight to the goal
            self.close_stop_flag = False
            robot_state = self.robot_pose.copy()
            st = robot_state.copy()

            if self.robot_goal is None:
                goal = np.array([6.0, 0.0])
            else:
                # goal = self.robot_goal[:2]
                st = self.robot_pose.copy()
                goal = np.zeros(2)
                goal[0] = st[0] + 1.0
                goal[1] = self.robot_goal[1]
                if np.linalg.norm(self.robot_goal[:2] - st[:2]) < 0.5:
                    goal = self.robot_goal[:2]

            if st[2] > 0.0:
                theta_a = st[2] - np.pi/2
            else:
                theta_a = st[2] + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])
            vec2goal = goal - st[:2]
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            radius = 0.5 * dist2goal / proj_len

            nominal_vel = self.open_space_velocity
            if st[2] > 0.0:
                ut = np.array([nominal_vel, -nominal_vel/radius])
            else:
                ut = np.array([nominal_vel,  nominal_vel/radius])
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1))

            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, nominal_vel+0.05, self.max_ang_vel, self.num_samples
            )
            traj_essemble = brne.traj_sim_essemble(
                np.tile(robot_state, reps=(self.num_samples,1)).T,
                ulist_essemble,
                self.dt
            )

            point_cloud_safety_idx, point_cloud_min_dist = brne.point_cloud_safety_idx(
                traj_essemble,
                self.scan_points.copy(),
                self.point_cloud_dist_threshold
            )
            point_cloud_safety_idx = np.array(point_cloud_safety_idx)

            # self.log('test point_cloud_safety_idx')
            # for idx in point_cloud_safety_idx:
            #     self.log('type: {}'.format(type(idx)))
            #     self.log('value: {}'.format(idx))

            num_safe_pc_points = 0
            if len(point_cloud_safety_idx.shape) == 2:
                num_safe_pc_points = len(point_cloud_safety_idx[0])
                point_cloud_safety_idx = point_cloud_safety_idx[0]
            else:
                num_safe_pc_points = len(point_cloud_safety_idx)
            # self.log('point_cloud_safety_idx: {}'.format(point_cloud_safety_idx))
            # self.log('num of safe trajs (point cloud): {}'.format(num_safe_pc_points))

            if num_safe_pc_points <= 1:
                unstuck_cmd = np.array([0.0, 0.0])
                if self.robot_pose[2] > 0.0:
                    unstuck_cmd[1] = -0.5
                else:
                    unstuck_cmd[1] = 0.5
                opt_cmds = np.tile(unstuck_cmd, reps=(self.plan_steps,1))
                self.log('free nav mode unsafe!!!')

                # self.log('pc_dist_threshold: {}'.format(self.point_cloud_dist_threshold))
                # self.log('#points from scan: {}'.format(len(self.scan_points)))
                # for _i in range(len(self.scan_points)):
                #     self.log('min dist[{}]: {}'.format(_i, point_cloud_min_dist[_i]))
                # for _i in range(len(point_cloud_safety_idx)):
                #     self.log('safety_idx[{}]: {}'.format(_i, point_cloud_safety_idx[_i]))

            else:
                self.log('exist safe free nav path!')

                safe_traj_essemble = traj_essemble[:,:,point_cloud_safety_idx]
                safe_ulist_essemble = ulist_essemble[:,point_cloud_safety_idx,:]

                # self.log('safe_traj_essemble.shape: {}'.format(safe_traj_essemble.shape))
                # self.log('safe_ulist_essemble.shape: {}'.format(safe_ulist_essemble.shape))

                end_pose_essemble = safe_traj_essemble[-1, 0:2, :].T
                dists2goal_essemble = np.linalg.norm(end_pose_essemble - goal, axis=1)
                opt_cmds = safe_ulist_essemble[:, np.argmin(dists2goal_essemble), :]
                # self.log('opt_cmds.shape: {}'.format(opt_cmds.shape))

            self.cmds = opt_cmds
            self.cmds_traj = self.cmd_tracker.sim_traj(robot_state, self.cmds)

            if self.cmd_counter > 0:
                self.cmd_counter = 0

            self.publish_trajectory(self.opt_traj_pub, self.cmds_traj[:,0], self.cmds_traj[:,1])
            # self.publish_markers([], [])

            if self.robot_goal is None:
                self.cmds = np.zeros((self.plan_steps, 2))
                self.cmds_traj = np.tile(robot_state, reps=(self.plan_steps,1))

        self.benchmark_pub.tock(log=False)

        # self.publish_trajectory(self.brne_traj_pub, self.cmds_traj[:, 0], self.cmds_traj[:, 1])

    def publish_trajectory(self, publisher, xs, ys):
        p = Path()
        p.header.frame_id = 'odom'

        for x, y in zip(xs, ys):
            pose = PoseStamped()
            pose.header = p.header
            pose.pose.position.x = x
            pose.pose.position.y = y

            p.poses.append(pose)

        publisher.publish(p)

    def publish_markers(self, xsa, ysa):
        ma = MarkerArray()
        for xs, ys in zip(xsa, ysa):
            m = Marker()
            m.header.frame_id = 'odom'
            m.ns = 'ped_traj'
            m.id = len(ma.markers)
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.1
            m.color.a = 1.0
            m.color.r = 0.96
            m.color.g = 0.50
            m.color.b = 0.19

            for x, y in zip(xs, ys):
                p = Point()
                p.x = x
                p.y = y
                m.points.append(p)

            ma.markers.append(m)
        self.marker_pub.publish(ma)

    def staircase_velocity(self, vel):
        """
        "Truncate" velocity like a staircase.
        """
        speed = np.linalg.norm(vel)

        if speed < 0.3:
            factor = 0.0
        elif speed < 0.6:
            factor = 0.3
        else:
            factor = speed

        new_vel = vel / speed * factor
        return new_vel


    def ped_cb(self, msg):
        """
        This is the pedestrian perception callback function.
        Everytime it receives new pedestrian information, it does the BRNE optimization to compute the optimal way
        points and convert the way points to optimal control commands, these commands will update the control buffer
        """

        self.last_ped_stamp = Time.from_msg(msg.header.stamp)

        # there should be an initialization flag, but in practice it does not really matter
        num_peds = len(msg.pedestrians)
        self.prev_ped_array = self.curr_ped_array.copy()
        # self.curr_ped_array = np.zeros((num_peds, 2))
        self.curr_ped_array = []

        if self.people_timeout_off:
            self.ped_msg_buffer = {}

        # force existing pedestrians' velocity to be zero in the timeout buffer
        for key in self.ped_msg_buffer:
            self.ped_msg_buffer[key][0].velocity.x = 0.0
            self.ped_msg_buffer[key][0].velocity.y = 0.0

        for i, ped in enumerate(msg.pedestrians):
            ped_pose = ped.pedestrian.pose

            if np.isnan(ped_pose.x) or np.isnan(ped_pose.y):
                self.log(f'Detect NAN on {ped.pedestrian.identifier} !!!')
                continue  # skip the pedestrian is reading is nan

            if self.frame2frame_enabled:
                ### F2F implementation
                f2f_vel = np.zeros(2)
                num_prev_peds = len(self.prev_ped_array)
                ped_position = np.array([ped_pose.x, ped_pose.y])
                # self.curr_ped_array[i] = ped_position.copy()
                self.curr_ped_array.append(ped_position.copy())

                if num_prev_peds > 0:
                    dists2prev = np.linalg.norm(self.prev_ped_array - ped_position, axis=1)
                    f2f_vel = ped_position - self.prev_ped_array[np.argmin(dists2prev)]
                    f2f_vel /= 0.034  # assuming pedestrian information is published at 33 hz

                    if self.staircase_truncation:
                        f2f_vel = self.staircase_velocity(f2f_vel)

                ped.pedestrian.velocity.x = f2f_vel[0]
                ped.pedestrian.velocity.y = f2f_vel[1]

                assert(ped.pedestrian.velocity.x == f2f_vel[0])
                assert(ped.pedestrian.velocity.y == f2f_vel[1])

            self.ped_msg_buffer[ped.pedestrian.identifier] = ped.pedestrian, self.last_ped_stamp

        self.curr_ped_array = np.array(self.curr_ped_array)

    def map_cb(self, msg):
        # we don't need to process map for now
        pass

    def goal_cb(self, msg):
        position = msg.pose.position
        self.robot_goal = np.array([position.x, position.y])
        self.params_pub.publish(self.params_msg)

        self.check_goal()

    def navigate_action(self, goal):
        self.goal_cb(goal.pose)
        self.from_action = True

        while self.robot_goal is not None:
            time.sleep(0.1)

        result = NavigateToPose.Result()
        return result

    def odom_cb(self, msg):
        # the odometry callback function updates the robot's current pose
        self.robot_pose = pose2d_transform(msg.pose.pose)

        if self.robot_goal is None:
            return

        self.check_goal()

    def check_goal(self):
        dist2goal = np.sqrt((self.robot_pose[0]-self.robot_goal[0])**2 + (self.robot_pose[1]-self.robot_goal[1])**2)
        # self.log(f'dist2goal: {dist2goal}')
        if dist2goal < 0.5:
            self.robot_goal = None

            if not self.from_action:
                # Minor hack: Instead of implementing an action server, we just publish a GoalStatus for bookkeeping
                g = GoalStatus()
                g.status = GoalStatus.STATUS_SUCCEEDED
                self.result_pub.publish(g)

    def timer_cb(self):
        """
        This the control callback function, it receives a fixed frequency timer signal,
        and publish the control command at the same frequency (10Hz here because dt = 0.1)
        """
        # self.log(f'cmd counter: {self.cmd_counter}')
        cmd = Twist()

        out_of_date = False
        if self.last_ped_stamp is None or self.last_ped_stamp + self.sensor_patience < self.get_clock().now():
            out_of_date = True
            if self.last_ped_stamp:
                self.get_logger().warn('Sensor data out of date!', throttle_duration_sec=5.0)
                self.get_logger().warn(f'  data: {self.last_ped_stamp}', throttle_duration_sec=5.0)
                self.get_logger().warn(f' clock: {self.get_clock().now()}', throttle_duration_sec=5.0)
            else:
                self.get_logger().warn('No sensor data received!', throttle_duration_sec=5.0)


        # here we get the command from the buffer, using self.cmd_counter to track the index
        #if self.cmd_counter >= self.plan_steps-1:
        if self.cmd_counter >= self.plan_steps-1 or self.robot_goal is None or out_of_date:
            cmd.linear.x = float(0.0)
            cmd.angular.z = float(0.0)
        else:
            cmd.linear.x = float(self.cmds[self.cmd_counter][0])
            cmd.angular.z = float(self.cmds[self.cmd_counter][1])
            self.cmd_counter += 1

        self.log(f'current control: [{cmd.linear.x}, {cmd.angular.z}]')
        self.cmd_vel_pub.publish(cmd)
        self.cmd_log.append([cmd.linear.x, cmd.angular.z])
        # np.savetxt('/home/msun/Downloads/brne_ros_cmd_log.txt', np.array(self.cmd_log))

        # self.cmd_counter += 1

    def _publish_log(self, s):
        msg = String()
        msg.data = s
        self.log_pub.publish(msg)

    def log(self, s):
        self._publish_log(s)
        # self.get_logger().info(s)


def main(args=None):
    rclpy.init(args=args)

    node = BrneNavRos()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
