import random
import numpy as np
import scipy as sp

# reproducibility
seeder = np.random.SeedSequence()
metaseed = seeder.entropy
rng = np.random.default_rng(seeder)

def r(a, x):
    if a == 0.5:
        return 1
    else:
        return ((2 * a - 1) ** -2) / (0.5 * ((2 * a - 1) ** -2 - 1) + x) - 1

def wf(alpha, f, m, n_list, max_gen):
    N = np.sum(n_list)
    N_deme = len(n_list)
    #deme_init = random.choices(range(N_deme), weights=n_list)[0]
    #state = [0] * N_deme
    #state[deme_init] = 1
    state = [int(n / 2) for n in n_list]
    update_mat = np.zeros([N_deme, N_deme])
    for i in range(N_deme):
        for j in range(N_deme):
            if i == j:
                update_mat[i, j] = 1 - m
            else:
                update_mat[i, j] = m * n_list[j] / (N - n_list[i])
    tfix = 0
    traj = []
    sums = np.zeros(max_gen)
    while tfix < max_gen and np.sum(state) != 0 and np.sum(state) != N:
        traj.append(state.copy())
        x = np.sum(state) / N
        tot_f = f * r(alpha, x)
        fit_vec = np.zeros(N_deme)
        for i in range(N_deme):
            x_i = state[i] / n_list[i]
            fit_vec[i] = tot_f * x_i / (tot_f * x_i + 1 - x_i)
        p_vec = update_mat @ fit_vec
        for i in range(N_deme):
            new = np.random.binomial(n_list[i], p_vec[i])
            state[i] = new
            sums[tfix] += np.round(new / n_list[i]) * n_list[i]
        tfix += 1
    return np.array(traj), sums