import torch
from tqdm import tqdm

def test(model, env, stat_keys, n_episodes=100):
    env_output = env.initial()
    agent_state = model.initial_state(batch_size=1)
    agent_state = tuple(s.to(device=model.device) for s in agent_state)

    stats = dict({k: [] for k in stat_keys})

    for episode in tqdm(range(n_episodes), desc='testing episode'):
        while True:
            with torch.no_grad():
                agent_output, agent_state = model(env_output, agent_state)
            env_output = env.step(agent_output['action'])
            if env_output['done']:
                break

        for k in stat_keys:
            stats[k].append(float(env_output[k].numpy()[0][0]))

    return stats
