import random
import pickle
import numpy as np

def is_essential_save(epoch, max_epochs, eval_frequency):
    essential_saves = [0.01, 0.1, 0.5, 0.97] # 0.1 stands for 10% of max_epochs
    essential_saves = [int(e * max_epochs) for e in essential_saves]
    window = 5 * eval_frequency
    for es in essential_saves:
        if epoch in range(es - window, es + window):
            return True
    return False

# ==============================================================================
# https://stackoverflow.com/questions/51918580/python-random-list-of-numbers-in-a-range-keeping-with-a-minimum-distance

def ranks(sample):
    """
    Return the ranks of each element in an integer sample.
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])

def sample_with_minimum_distance(n=40, k=4, d=10):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """
    sample = random.sample(range(n-(k-1)*(d-1)), k)
    return [s + (d-1)*r for s, r in zip(sample, ranks(sample))]
# ==============================================================================


def read_habitat_data(data_path):
    print('loading %s ...' % data_path)

    # Merge trajectories
    data = pickle.load(open(data_path, 'rb'))
    n_trajectories = len(data['reward'])
    data['obs'] = np.concatenate(data['obs'])
    data['action'] = np.concatenate(data['action'])
    data['reward'] = np.concatenate(data['reward'])
    data['done'] = np.concatenate(data['done'])
    data['true_state'] = np.concatenate(data['true_state'])

    n_samples = len(data['reward'])
    print('  ', '%d trajectories for a total of %d samples' % (n_trajectories, n_samples))
    print('  ', 'avg. return is', data['reward'].sum() / n_trajectories)

    return data
