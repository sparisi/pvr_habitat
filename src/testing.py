import traceback
import torch
import numpy as np
from tqdm import tqdm
from src.gym_wrappers import make_gym_env

STOP = 'STOP'


def is_episode_successful(env_id, scs_sum):
    if 'Habitat' in env_id:
        n = 0 # only last step is successful
    elif env_id == 'HMS-pen-v0':
        n = 20 # github.com/vikashplus/mj_envs/blob/stable/mj_envs/envs/hand_manipulation_suite/pen_v0.py#L140
    elif env_id == 'HMS-relocate-v0':
        n = 25 # github.com/vikashplus/mj_envs/blob/stable/mj_envs/envs/hand_manipulation_suite/relocate_v0.py#L115
    else:
        return False
    return scs_sum > n


def online_test(queue, model, embedding_model, shared_stats, flags):
    while True: # keep process alive for testing
        epoch, env_id = queue.get()
        if env_id == STOP:
            queue.task_done()
            break

        try:
            env = make_gym_env(env_id, flags, embedding_model)
            shared_stats.update({env_id: {k: [] for k in ['episode_return', 'episode_success']}})
            returns = []
            successes = []
            for episode in tqdm(range(flags.num_episodes_test), desc=f'{env_id} @ {epoch}', mininterval=5, leave=False):
                obs = env.reset()
                agent_state = model.initial_state(batch_size=1)
                rwd_sum = 0.
                scs_sum = 0
                infos = []
                done = False
                while not done:
                    with torch.no_grad():
                        env_state = dict(
                            obs=torch.as_tensor(obs.copy()).view((1, 1) + obs.shape),
                            done=torch.as_tensor(done).view(1, 1),
                        )
                        agent_output, agent_state = model(env_state, agent_state)
                        if env.action_space.shape == ():
                            action = agent_output['action'].item()
                        else:
                            action = agent_output['action'].flatten().detach().cpu().numpy()
                    obs, rwd, done, info = env.step(action)
                    rwd_sum += rwd
                    try: scs_sum += 1. * info['success']
                    except: pass # not all envs have this info
                returns.append(rwd_sum)
                successes.append(is_episode_successful(env_id, scs_sum))
            shared_stats.update({env_id: {'episode_return': returns, 'episode_success': successes}})
            print(f'   ___ {epoch} | {env_id} : return {np.mean(returns)}, success {np.mean(successes)}')

        except:
            traceback.print_exc()
            break

        finally:
            try: env.close()
            except: pass
            queue.task_done() # notify parent that testing is done for requested (epoch, env_id)
