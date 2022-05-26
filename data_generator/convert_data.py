import pickle5 as pickle
import numpy as np


### DMC trajectories
# For each env there are 2 pickles:
# - one with full trajectories with image obs,
# - and one with true_state obs.
file_load = [
    'dmc_cheetah_run-v1.pickle',
    'dmc_walker_stand-v1.pickle',
    'dmc_finger_spin-v1.pickle',
    'dmc_walker_walk-v1.pickle',
    'dmc_reacher_easy-v1.pickle'
]
for f in file_load:
    new_data = dict()
    data = pickle.load(open('dmc_img/' + f, 'rb'))
    new_data.update({'obs': [t['images'] for t in data]})
    new_data.update({'action': [t['actions'] for t in data]})
    new_data.update({'done':
        [np.concatenate((np.full((t['actions'].shape[0]-1,), False), np.array([True]))) for t in data]
    }) # these are infinite-horizon tasks, so all but last step must have done=False
    data = pickle.load(open('dmc_states/' + f, 'rb'))
    new_data.update({'true_state': [t['observations'] for t in data[:100]]}) # only first 100 traj are used in the other pickles
    file_save = f.replace('dmc', 'DMC').replace('-v1', '').replace('_', '-')
    pickle.dump(new_data, open(file_save, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


### HMS trajectories
file_load = [
    'relocate-v0.pickle',
    'pen-v0.pickle',
]
for f in file_load:
    data = pickle.load(open(f, 'rb'))
    new_data = dict()
    new_data.update({'obs': [t['images'] for t in data]})
    new_data.update({'action': [t['actions'] for t in data]})
    new_data.update({'done':
        [np.concatenate((np.full((t['actions'].shape[0]-1,), False), np.array([True]))) for t in data]
    }) # these are infinite-horizon tasks, so all but last step must have done=False
    new_data.update({'true_state': [t['state_obs'] for t in data]})
    file_save = 'HMS-' + f
    pickle.dump(new_data, open(file_save, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
