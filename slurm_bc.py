import argparse
import datetime
import itertools
import pprint
import os
import submitit
import pickle
from collections import defaultdict

from src.arguments import parser as runner_parser
from main_bc_1 import run as runner_main_random
from main_bc_2 import run as runner_main_pvr

os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--partition', type=str, default='learnfair',
                    choices=['learnfair', 'devlab', 'prioritylab'])

max_frames = defaultdict(lambda: 200000000)
max_frames.update({
    'HabitatImageNav-apartment_0': 200000000,
    'HabitatPointNav-apartment_0': 2000000,
})

# key => k; some_key => sk
def make_prefix(key):
    tokens = key.split('_')
    return ''.join(w[0] for w in tokens)


def expand_args(params):
    sweep_args = {k: v for k, v in params.items() if isinstance(v, list)}
    # sweep :: [{arg1: val1, arg2: val1}, {arg1: val2, arg2: val2}, ...]
    sweep = [
        dict(zip(sweep_args.keys(), vs))
        for vs in itertools.product(*sweep_args.values())
    ]
    expanded = []
    for swargs in sweep:
        new_args = {**params, **swargs}  # shallow merge
        new_args['xpid'] = '--'.join(
            [f'{make_prefix(k)}={v}' for k, v in swargs.items()])
        expanded.append(new_args)

    for exp in expanded:
        exp['max_frames'] = max_frames[exp['env']]

    return expanded


args_grid = dict(
    env=[
        'HabitatImageNav-apartment_0,HabitatImageNav-frl_apartment_0,HabitatImageNav-office_0,HabitatImageNav-room_0,HabitatImageNav-hotel_0',
    ],
    to_env=[
        'HabitatImageNav-apartment_0',
        'HabitatImageNav-frl_apartment_0',
        'HabitatImageNav-office_0',
        'HabitatImageNav-room_0',
        'HabitatImageNav-hotel_0',
    ],
    embedding_name=[
        'true_state',
        #
        'moco_croponly_places_uber_345',
        'moco_croponly_uber_345',
        'moco_croponly_places_uber_35',
        'moco_croponly_uber_35',
        'moco_croponly_places_uber_34',
        'moco_croponly_uber_34',
        'moco_croponly_places_uber_45',
        'moco_croponly_uber_45',
        # 'moco_aug_places_uber_345',
        # 'moco_aug_uber_345',
        # 'moco_aug_places_uber_35',
        # 'moco_aug_uber_35',
        # 'moco_aug_places_uber_34',
        # 'moco_aug_uber_34',
        # 'moco_aug_places_uber_45',
        # 'moco_aug_uber_45',
        #
        # 'moco_croponly_habitat',
        'moco_aug_habitat',
        #
        'moco_croponly_places_l4',
        'moco_croponly_places_l3',
        'moco_croponly_places',
        'moco_croponly_l3',
        'moco_croponly_l4',
        'moco_croponly',
        #
        'moco_coloronly',
        #
        'moco_aug_places_l3',
        'moco_aug_places_l4',
        'moco_aug_places',
        'moco_aug_l4',
        'moco_aug_l3',
        'moco_aug',
        #
        # 'demy',
        # 'maskrcnn_l3',
        'clip_rn50',
        'clip_vit',
        'mae_base',
        'mae_large',
        #
        'resnet34',
        'resnet50',
        'resnet50_places',
        'resnet50_l4',
        'resnet50_l3',
        'resnet50_places_l4',
        'resnet50_places_l3',
        #
        'random',
    ],
    save_path=['latest'],
    max_frames=[0],
    run_id=[1,2,3,4,5,6,7,8,9,10],
    unroll_length=[100],
    batch_size=[16],
    learning_rate=[0.0001],
    n_episodes_test=[50],
    eval_frequency=[200],
)


# NOTE params is a shallow merge, so do not reuse values
def make_command(params, unique_id):
    # creating cmd-like params
    params = itertools.chain(*[('--%s' % k, str(v))
                               for k, v in params.items()])
    return list(params)


args = parser.parse_args()
args_grid = expand_args(args_grid)
print(f"Submitting {len(args_grid)} jobs to Slurm...")

uid = datetime.datetime.now().strftime('%H-%M-%S-%f')
job_index = 0

for run_args in args_grid:
    flags = runner_parser.parse_args(make_command(run_args, uid))

    # Skip transfer runs
    if not (flags.to_env in flags.env):
        print('skipping', run_args)
        print()
        continue

    # Check if run was already done, and if so skip it
    save_path = os.path.join(flags.save_path, \
                flags.env + '_em' + \
                flags.embedding_name + '_s' + \
                str(flags.run_id) + '_' + \
                flags.to_env)
    if os.path.isfile(save_path + '.pickle'):
#        continue
        stats = pickle.load(open(save_path + '.pickle', 'rb'))
        if stats[flags.to_env]['frames'][-1] >= flags.max_frames - flags.unroll_length * flags.batch_size:
            print('skipping', run_args)
            print()
            continue

    flags.essential_save_only = True

    if flags.embedding_name != 'true_state':
        flags.batch_norm = True

    job_index += 1

    print('########## Job {:>4}/{} ##########\nFlags: {}'.format(
        job_index, len(args_grid), flags))

    if args.local:
        executor_cls = submitit.LocalExecutor
    else:
        executor_cls = submitit.SlurmExecutor

    executor = executor_cls(folder='./out/')

    partition = args.partition
    if args.debug:
        partition = 'devlab'

    num_scenes = len(flags.env.split(','))
    mem = 16 * num_scenes * (2 if 'uber' in flags.embedding_name else 1)
    executor.update_parameters(
        partition=partition,
        comment='icml_27_01',
        time=4319,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name='%s-%s-%s-%s' % ('bc', run_args['embedding_name'], run_args['env'], run_args['to_env']),
        mem=str(mem)+"GB",
        cpus_per_task=5,
        num_gpus=1,
        constraint='pascal',
    )

    print('Sending to slurm... ', end='')
    if flags.embedding_name == 'random':
        job = executor.submit(runner_main_random, flags)
    else:
        job = executor.submit(runner_main_pvr, flags)
    print('Submitted with job id: ', job.job_id)

    if args.debug:
        print('Only running one job on devfair for debugging...')
        import sys
        sys.exit(0)
