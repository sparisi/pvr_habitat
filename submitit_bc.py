import argparse
import itertools
import os
import submitit
import pickle5 as pickle
import copy
import warnings

from main_bc import run as runner_main
from main_bc import parser as runner_parser

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./results_fast/')
parser.add_argument('--log_path', type=str, default='./out/')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--partition', type=str, default='learnfair')
parser.add_argument('--dist_backend', type=str, default='nccl')
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--gpus_per_node', type=int, default=1)
parser.add_argument('--cpus_per_task', type=int, default=8)


class Worker:
    """
    Worker called by SLURM.
    It sets all arguments for distributed training and then calls the main
    training function (runner_main).
    """
    def __call__(self, origargs):
        import numpy as np
        import random
        import torch
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

        args = copy.deepcopy(origargs)
        np.set_printoptions(precision=3)

        socket_name = os.popen(
            "ip r | grep default | awk '{print $5}'"
        ).read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        os.environ["NCCL_SOCKET_IFNAME"] = socket_name

        job_env = submitit.JobEnvironment()
        args.rank = job_env.global_rank
        args.port = 21992
        args.dist_url = f'tcp://{job_env.hostnames[0]}:{args.port}'
        print('Using url {} with {} backend.'.format(args.dist_url, args.dist_backend))

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        runner_main(args)

    def checkpoint(self, *args,
                   **kwargs) -> submitit.helpers.DelayedSubmission:
        """
        Auto-resume jobs.
        """
        return submitit.helpers.DelayedSubmission(
            self, *args, **kwargs)


################################################################################


def load_jobs(N=1000, end_after="$(date +%Y-%m-%d-%H:%M)"):
    jobs = (os.popen(
        f'sacct -u $USER --format="JobID,JobName,Partition,State,End,Comment" '
        f'-X -P -S "{end_after}" | tail -n {N}').read().split("\n"))
    jobs_parsed = []
    for line in jobs:
        row = line.strip().split("|")
        if len(row) != 6:
            continue
        if row[0] == "JobID":
            continue
        job_id_raw, name, partition, status, end, comment = row
        job_id_comp = job_id_raw.strip().split("_")
        job_id = int(job_id_comp[0])
        try:
            if len(job_id_comp) == 2:
                sort_key = (end, job_id, int(job_id_comp[1]))
            else:
                sort_key = (end, job_id, 0)
        except ValueError:
            print("Error parsing job: ", job_id)
            continue
        jobs_parsed.append(
            [job_id_raw, name, partition, status, end, comment, sort_key])
    jobs_parsed = sorted(jobs_parsed, key=lambda el: el[-1])
    return jobs_parsed


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
        ('HabitatImageNav-apartment_0,'
        'HabitatImageNav-frl_apartment_0,'
        'HabitatImageNav-office_0,'
        'HabitatImageNav-room_0,'
        'HabitatImageNav-hotel_0') # this is one string: 5 envs separated by commas
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
    seed=[1,2,3,4,5],
    unroll_length=[100],
    batch_size=[16],
    # train_embedding=[True,False],
)

args = parser.parse_args()
args_grid = expand_args(args_grid)
print(f"Submitting {len(args_grid)} jobs to Slurm...")

job_index = 0
jobdets = load_jobs()
jobnames = [j[1] for j in jobdets]

for run_args in args_grid:
    flags = runner_parser.parse_args(make_command(run_args))

    flags.save_path = args.save_path
    save_path = os.path.join(flags.save_path, flags.xpid)
    job_name = 'bc_' + flags.xpid

    # Skip running jobs (Slurm automatically replaces some chars with underscore)
    if job_name.replace(',', '_').replace('-', '_') in jobnames:
        print(' _ Skipping (already running) : {}\n'.format(job_name))
        continue

    # Skip completed jobs
    if os.path.isfile(save_path + '.pickle'):
        stats = pickle.load(open(save_path + '.pickle', 'rb'))
        if stats['epoch'][-1] >= flags.epochs:
            print(' _ Skipping (already done) : {}\n'.format(job_name))
            continue

    if flags.embedding_name != 'true_state':
        flags.batch_norm = True
    flags.world_size = args.gpus_per_node * args.nodes
    flags.dist_backend = args.dist_backend
    flags.cpus_per_task = args.cpus_per_task

    job_index += 1

    print('# Job {}/{} : {}'.format(job_index, len(args_grid), job_name))

    # slurm_max_num_timeout: how many times the job will be checkpointed and requeued if timed out or preempted
    executor = submitit.AutoExecutor(folder=args.log_path, slurm_max_num_timeout=100)
    executor.update_parameters(
        name=job_name,
        slurm_partition='devlab' if args.debug else args.partition,
        slurm_comment='this_is_a_comment',
        timeout_min=1440, # max job time in minutes
        nodes=args.nodes,
        tasks_per_node=args.gpus_per_node,
        gpus_per_node=args.gpus_per_node,
        cpus_per_task=args.cpus_per_task,
        mem_gb=512,
    )
    # A note regarding `mem_gb`. It is the memory per node in GB, so if you use M
    # GPU per node then you will load your data M times. Request memory accordingly.

    print('Sending to slurm... ', end='')
    job = executor.submit(Worker(), flags)
    print('Submitted with job id: ', job.job_id, '\n')

    if args.debug:
        print(' STOPPING. Running only one job on devfair for debugging...')
        import sys
        sys.exit(0)
