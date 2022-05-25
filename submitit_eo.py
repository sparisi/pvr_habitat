import argparse
import itertools
import os
import submitit
import pickle5 as pickle

from data_generator.save_embedded_obs import run as runner_main
from data_generator.save_embedded_obs import parser as runner_parser

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--partition', type=str, default='learnfair')
parser.add_argument('--log_path', type=str, default='./out/')

################################################################################


### The experiment unique id (xpid) is generated using the arguments defined
### in args_grid by taking the initials of the arguments and their value.

# key => k; some_key => sk
def make_prefix(key):
    return ''.join(w[0] for w in key.split('_'))

# {key: v1, some_key: v2} => kv1--skv2
def dict_to_xpid(d):
    return '_'.join(
        [f'{make_prefix(k)}{v}' for k, v in d.items()])

# Generates all combinations of parameters defined in args_grid and merges
# them with default parameters retrieved from the parser
def expand_args(params):
    # sweep :: [{arg1: val1, arg2: val1}, {arg1: val2, arg2: val2}, ...]
    sweep_args = {k: v for k, v in params.items() if isinstance(v, list)}
    sweep = [
        dict(zip(sweep_args.keys(), vs))
        for vs in itertools.product(*sweep_args.values())
    ]
    expanded = []
    for swargs in sweep:
        new_args = {**params, **swargs} # shallow merge
        new_args['xpid'] = dict_to_xpid(swargs)
        expanded.append(new_args)

    return expanded

# Makes cmd-like params
def make_command(params):
    return [
            ('--%s%s' % (k, '=' + str(v) if not (type(v) == bool and v) else '')) # args like --train=True become just --train
            for k, v in params.items()
            if not (type(v) == bool and not v) # args like --train=False are skipped,
           ]


################################################################################


args_grid = dict(
    env=[
        'HabitatImageNav-apartment_0',
        'HabitatImageNav-frl_apartment_0',
        'HabitatImageNav-office_0',
        'HabitatImageNav-room_0',
        'HabitatImageNav-hotel_0',
        'HMS-pen-v0',
        'HMS-relocate-v0',
    ],
    embedding_name=[
        'moco_croponly_places_uber_345',
        'moco_croponly_uber_345',
        'moco_croponly_places_uber_35',
        'moco_croponly_uber_35',
        'moco_croponly_places_uber_34',
        'moco_croponly_uber_34',
        'moco_croponly_places_uber_45',
        'moco_croponly_uber_45',
        'moco_aug_places_uber_345',
        'moco_aug_uber_345',
        'moco_aug_places_uber_35',
        'moco_aug_uber_35',
        'moco_aug_places_uber_34',
        'moco_aug_uber_34',
        'moco_aug_places_uber_45',
        'moco_aug_uber_45',
        #
        'moco_croponly_habitat',
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
)

args = parser.parse_args()
args_grid = expand_args(args_grid)
print(f"Submitting {len(args_grid)} jobs to Slurm...")

job_index = 0

for run_args in args_grid:
    flags = runner_parser.parse_args(make_command(run_args))
    job_name = 'eo_' + flags.xpid

    # Check if run was already done, and if so skip it
    save_name = os.path.join(flags.data_path,
                             flags.env + '_' +
                             flags.embedding_name + '.pickle')
    if os.path.isfile(save_name):
        print(' _ Skipping (already done) : {}\n'.format(job_name))
        continue

    job_index += 1

    print('# Job {}/{} : {}'.format(job_index, len(args_grid), job_name))

    executor = submitit.AutoExecutor(folder=args.log_path, slurm_max_num_timeout=100)
    executor.update_parameters(
        name=job_name,
        slurm_partition='devlab' if args.debug else args.partition,
        slurm_comment='this_is_a_comment',
        timeout_min=1440,
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=8,
        mem_gb=64,
    )

    print('Sending to slurm... ', end='')
    job = executor.submit(runner_main, flags)
    print('Submitted with job id: ', job.job_id, '\n')

    if args.debug:
        print(' STOPPING. Running only one job on devfair for debugging...')
        import sys
        sys.exit(0)
