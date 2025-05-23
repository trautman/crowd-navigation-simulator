########################################################## FUNCTIONS
import numpy as np
import numba as nb

# import os
# os.environ['NUMBA_NUM_THREADS'] = '12'
# nb.config.NUMBA_NUM_THREADS = 4

rng = np.random.default_rng(1)


"""
BRNE parameters:
kernel_a1: Control the "straightness" of trajectory samples. Larger the value is, less straight the trajectory sample will be.
kernel_a2: Control the "width/spreadness" of trajectory samples. Larger the value is, more spread the trajectory samples are.
cost_a1: Control the safety zone, smaller the value is, more conversative the robot will be.
cost_a2: Control the safety zone, larger the value is, more conservative the robot will be.
cost_a3: Control the safety penalty weight, larger the value is, more conservative the robot will be.
"""
# kernel_a1 = 0.5
# kernel_a2 = 0.2
# cost_a1 = 8.0
# cost_a2 = 1.0
# cost_a3 = 20.0


@nb.jit(nopython=True, parallel=True)
def get_kernel_mat_nb(t1list, t2list, kernel_a1, kernel_a2):
    mat = np.zeros((t1list.shape[0], t2list.shape[0]))
    for i in nb.prange(t1list.shape[0]):
        for j in nb.prange(t2list.shape[0]):
            mat[i][j] = np.exp(-kernel_a1 * (t1list[i] - t2list[j]) ** 2) * kernel_a2
    return mat.T


def mvn_sample_normal(_num_samples, _tsteps, _Lmat):
    init_samples = rng.standard_normal(size=(_tsteps, _num_samples))
    new_samples = _Lmat @ init_samples
    return new_samples.T


def get_min_dist(x_trajs, y_trajs):
    dx_trajs = x_trajs[:, np.newaxis, :] - x_trajs
    dy_trajs = y_trajs[:, np.newaxis, :] - y_trajs
    dist_trajs = np.sqrt(dx_trajs ** 2 + dy_trajs ** 2)

    min_dist_trajs = np.min(dist_trajs, axis=2)
    min_dist_trajs += np.eye(min_dist_trajs.shape[0]) * 1e6
    min_dist = np.min(min_dist_trajs)

    return min_dist


@nb.njit  # ('float64[:, :](float64[:, :])')
def cholesky_numba(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = 0
            for k in range(j):
                s += L[i][k] * L[j][k]

            if (i == j):
                L[i][j] = (A[i][i] - s) ** 0.5
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


@nb.jit(nopython=True)
def get_Lmat_nb(train_ts, test_ts, train_noise, kernel_a1, kernel_a2):
    covmat_11 = get_kernel_mat_nb(train_ts, train_ts, kernel_a1, kernel_a2)
    covmat_11 += np.diag(train_noise)
    covmat_12 = get_kernel_mat_nb(test_ts, train_ts, kernel_a1, kernel_a2).T
    covmat_22 = get_kernel_mat_nb(test_ts, test_ts, kernel_a1, kernel_a2)
    cov_mat = covmat_22 - covmat_12 @ np.linalg.inv(covmat_11) @ covmat_12.T
    cov_mat += np.eye(test_ts.shape[0]) * 1e-06
    return cholesky_numba(cov_mat), cov_mat


@nb.jit(nopython=True, parallel=True)
def costs_nb(trajs_x, trajs_y, num_agents, num_pts, tsteps, cost_a1, cost_a2, cost_a3):
    vals = np.zeros((num_agents * num_pts, num_agents * num_pts))
    for i in nb.prange(num_pts * num_agents):
        traj_xi = trajs_x[i]
        traj_yi = trajs_y[i]
        for j in nb.prange(num_pts * num_agents):
            traj_xj = trajs_x[j]
            traj_yj = trajs_y[j]
            traj_costs = np.zeros(tsteps)
            for t in nb.prange(tsteps):
                dist = (traj_xi[t] - traj_xj[t]) ** 2 + (traj_yi[t] - traj_yj[t]) ** 2
                traj_costs[t] = 2 - 2 / (1.0 + np.exp(-cost_a1 * np.power(dist, cost_a2)))
            vals[i, j] = np.max(traj_costs) * cost_a3
    return vals


@nb.jit(nopython=True, parallel=True)
def weights_update_nb(all_costs, old_weights, index_table, all_pt_index, num_agents, num_pts):
    weights = old_weights.copy()
    for i in range(num_agents):
        row = index_table[i]
        # other_pt_index = all_pt_index[row[1:], :].ravel()

        for j1 in nb.prange(num_pts):
            cost1 = 0.0
            idx1 = all_pt_index[row[0], j1]
            for i2 in nb.prange(num_agents - 1):
                for j2 in range(num_pts):
                    idx2 = all_pt_index[row[i2 + 1], j2]
                    cost1 += all_costs[idx1, idx2] * weights[row[i2 + 1], j2]
            cost1 /= (num_agents - 1) * num_pts
            weights[i, j1] = np.exp(-1.0 * cost1)
        weights[i] /= np.mean(weights[i])
    return weights


@nb.jit(nopython=True, parallel=True)
def get_index_table(num_agents):
    index_table = np.zeros((num_agents, num_agents))
    for i in nb.prange(num_agents):
        index_table[i][0] = i
        idx = 1
        for j in range(num_agents):
            if i == j:
                continue
            index_table[i][idx] = j
            idx += 1
    return index_table

@nb.jit(nopython=True, parallel=True)
def coll_beck(trajs_y, y_min, y_max):
    lower_mask = trajs_y > y_min
    upper_mask = trajs_y < y_max
    return lower_mask * upper_mask

def brne_nav(xtraj_samples, ytraj_samples, num_agents, tsteps, num_pts, cost_a1, cost_a2, cost_a3, ped_sample_scale, y_min, y_max):
    index_table = get_index_table(num_agents).astype(int)
    all_pt_index = np.arange(num_agents * num_pts).reshape(num_agents, num_pts)

    weights = np.ones((num_agents, num_pts))

    all_costs = costs_nb(xtraj_samples, ytraj_samples, num_agents, num_pts, tsteps, cost_a1, cost_a2, cost_a3)

    coll_mask = coll_beck(ytraj_samples[0:num_pts], y_min, y_max).all(axis=1).astype(float)

    for iter_num in range(11):
        weights = weights_update_nb(all_costs, weights, index_table, all_pt_index, num_agents, num_pts)

    # —— enforce lateral corridor on the robot’s x‐trajectories ——
    # the first num_pts rows of xtraj_samples are the robot’s rollouts
    robot_x_trajs = xtraj_samples[0:num_pts]                        # shape (num_pts, tsteps)

    # build a mask: True only if every time‐step stays within [x_min, x_max]
    coll_mask = coll_beck(robot_x_trajs, y_min, y_max).all(axis=1).astype(float)

    # zero‐out any rollout that leaves the corridor
    masked_weights = weights[0] * coll_mask
    mean_w = masked_weights.mean()

    if mean_w > 0.0:
        # renormalize so the surviving rollouts still average to 1.0
        masked_weights /= mean_w
        weights[0] = masked_weights.copy()
    # else: leave weights[0] untouched to avoid divide‐by‐zero/NaN
 
    # # —— enforce lateral corridor on the robot’s x-trajs —— 
    # # first num_pts rows of xtraj_samples are the robot’s candidates
    # robot_x_trajs = xtraj_samples[0:num_pts]                   # shape (num_pts, tsteps)

    # # mask is True only if the entire rollout stays within [x_min, x_max]
    # per_step = coll_beck(robot_x_trajs, y_min, y_max)         # (num_pts, tsteps)
    # coll_mask = per_step.all(axis=1).astype(float)            # (num_pts,)

    # # zero out any rollout that leaves the corridor
    # masked_weights = weights[0] * coll_mask
    # mean_w = masked_weights.mean()

    # if mean_w > 0.0:
    #     masked_weights /= mean_w
    #     weights[0] = masked_weights.copy()


    return weights

# @nb.jit(nopython=True)
def get_min_dist(x_trajs, y_trajs):
    dx_trajs = x_trajs[:,np.newaxis,:] - x_trajs
    dy_trajs = y_trajs[:,np.newaxis,:] - y_trajs
    dist_trajs = np.sqrt(dx_trajs**2 + dy_trajs**2)

    min_dist_trajs = np.min(dist_trajs, axis=2)
    min_dist_trajs += np.eye(min_dist_trajs.shape[0]) * 1e6
    min_dist = np.min(min_dist_trajs[0])
    # print('min dist verify: ', np.min(min_dist_trajs[1]))

    return min_dist

##########################################################################################################
def dyn(st, ut):
    sdot = np.array([
        ut[0] * np.cos(st[2]),
        ut[0] * np.sin(st[2]),
        ut[1]
    ])
    return sdot

def dyn_step(st, ut, dt):
    k1 = dt * dyn(st, ut)
    k2 = dt * dyn(st + k1/2.0, ut)
    k3 = dt * dyn(st + k2/2.0, ut)
    k4 = dt * dyn(st + k3, ut)

    return st + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

def traj_sim(st, ulist, dt):
    tsteps = len(ulist)
    traj = np.zeros((tsteps, 3))
    for t in range(tsteps):
        st = dyn_step(st, ulist[t])
        traj[t] = st.copy()
    return traj

def traj_sim_essemble(st, ulist, dt):
    """
    st.shape = (3, num_samples)
    ulist = (tsteps, num_samples, 2)
    """
    tsteps = ulist.shape[0]
    num_samples = ulist.shape[1]
    assert st.shape[1] == ulist.shape[1]

    traj = np.zeros((tsteps, 3, num_samples))

    for t in range(tsteps):
        st = dyn_step(st, ulist[t].T, dt)
        traj[t] = st.copy()

    return traj

def get_ulist_essemble(ulist, u1_max, u2_max, num_samples):
    num_essembles_per_dim = int(np.sqrt(num_samples))
    num_essembles_per_dim_u1 = int(num_essembles_per_dim * 2)
    num_essembles_per_dim_u2 = int(num_essembles_per_dim / 2)

    u1_offset = np.minimum(
        ulist[:,0].min(),
        u1_max-ulist[:,0].max()
    )
    u2_offset = np.minimum(
        u2_max+ulist[:,1].min(),
        u2_max-ulist[:,1].max()
    )
    # print('u offsets: ', u1_offset, u2_offset)

    # u1_offset = np.minimum(ulist[0][0], 0.2)
    # u2_offset = 1.2

    # ### grid approach
    u1_perturbs, u2_perturbs = np.meshgrid(
       np.linspace(-u1_offset, u1_offset, num_essembles_per_dim_u1),
       np.linspace(-u2_offset, u2_offset, num_essembles_per_dim_u2)
    )

    ### gaussian sampling
    # u1_perturbs = np.random.normal(size=(num_samples,))
    # u1_perturbs /= -u1_perturbs.min() / (ulist[0][0] - 0.0)
    # u2_perturbs = np.random.normal(size=(num_samples,))
    # u2_perturbs /= u2_perturbs.max()

    u_perturbs = np.array([
        u1_perturbs.ravel(), u2_perturbs.ravel()
    ]).T

    ulist_essemeble = ulist[:,np.newaxis] + u_perturbs
    # ulist_essemeble[:,:,0] = np.clip(ulist_essemeble[:,:,0], 0.0, u1_max)
    # ulist_essemeble[:,:,1] = np.clip(ulist_essemeble[:,:,1], -u2_max, u2_max)

    assert ulist_essemeble.shape[0] == ulist.shape[0]
    assert ulist_essemeble.shape[1] == u_perturbs.shape[0]
    assert ulist_essemeble.shape[2] == ulist.shape[1]

    return ulist_essemeble

###################################################################

def point_cloud_safety_idx(traj_essemble, point_cloud, threshold):
    traj_essemble_x = traj_essemble[:,0,:]
    traj_essemble_y = traj_essemble[:,1,:]
    point_cloud_x = point_cloud[:,0]
    point_cloud_y = point_cloud[:,1]
    dist_essemble = np.sqrt(
        (traj_essemble_x[:,:,np.newaxis] - point_cloud_x) ** 2 + \
        (traj_essemble_y[:,:,np.newaxis] - point_cloud_y) ** 2
    )
    min_dist_essemble = np.min(np.min(dist_essemble, axis=2), axis=0)

    safe_idx = np.where(min_dist_essemble > threshold)
    return safe_idx, min_dist_essemble
