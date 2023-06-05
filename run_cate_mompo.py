from agents.mpo_humanoid import GaussianMOMPO, BehaviorGaussianMPO
from envs.deep_sea_treasure import DeepSeaTreasure

import argparse
import os
from collections import namedtuple
import numpy as np
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--logdir', default='./logs', type=str, help='base directory to save log')
    parse.add_argument('--model', default='', type=str, help='pretrained model path')
    parse.add_argument('--env', default='DeepSeaTreasure', type=str, help='environment name')
    parse.add_argument('--beta', default=0.001, type=int, help='KL constraint on the change of policy')
    parse.add_argument('--gamma', default=0.999, type=int, help='discount factor')
    parse.add_argument('--epsilons', default='0.005,0.01', type=str, help='epsilon of different objective')
    parse.add_argument('--test_only', default=False, action='store_true')
    parse.add_argument('--train_iter', default=10000, type=int, help='training iterations')
    parse.add_argument('--test_iter', default=10, type=int, help='testing iterations')
    parse.add_argument('--device', default='cpu', type=str, help='device')
    parse.add_argument('--multiprocess', default=1, type=int, help='how many process for asynchronous actor')

    args = parse.parse_args()

    return args


def SingleTrain(agent: GaussianMOMPO, args, k):
    Transition = namedtuple(
        'Transition', ('state', 'action', 'log_prob', 'reward', 'done'))
    env = args.env
    device = args.device

    for i in range(args.train_iter): 
        state = env.reset()
        t = 0
        episode_reward = np.zeros((k))
        trajectory = []
        while True:
            action, log_prob = agent.select_action(torch.tensor(state, dtype=torch.float, device=device))
            next_state, reward, done = env.step(action)
            trajectory.append(Transition(state, [action], reward, [log_prob], [int(done)]))
            state = next_state
            episode_reward += reward
            t += 1
            if done:
                break

        print(f"Episode: {i}, length: {t} ")
        for i in range(episode_reward.shape[0]):
            print(f'reward{i}: {episode_reward[i]} ', end='')
        print()

        agent._replay_buffer.push(trajectory)
    
        agent.update()

    agent.save(args.logdir)


def MultiTrain(shared_actor, args, k, state_dim, action_dim):
    Transition = namedtuple(
        'Transition', ('state', 'action', 'log_prob', 'reward', 'done'))
    env = args.env
    device = args.device

    # asyncronous actor
    agent = BehaviorGaussianMPO(state_dim, action_dim)

    for i in range(args.train_iter): 
        # load new actor
        print('loading new actor...')
        agent._actor.load_state_dict(shared_actor.state_dict())
        print('finish')
        time.sleep(100000)

        state = env.reset()
        t = 0
        episode_reward = np.zeros((k))
        trajectory = []
        while True:
            action, log_prob = agent.select_action(torch.tensor(state, device=device))
            next_state, reward, done = env.step(action)
            trajectory.append(Transition(state, [action], reward, [log_prob], [int(done)]))
            state = next_state
            episode_reward += reward
            t += 1
            if done:
                break

        print(f"Episode: {i}, length: {t} ")
        for i in range(episode_reward.shape[0]):
            print(f'reward{i}: {episode_reward[i]} ', end='')
        print()

        # replay_buffer.push(trajectory)


def Learner(agent: GaussianMOMPO, ps):
    all_ps_finish = False
    t = 0
    while not all_ps_finish:
        t += 1
        agent.update(t)
        all_ps_finish = True
        for p in ps:
            if p.is_alive():
                all_ps_finish = False
                break


def test(agent: GaussianMOMPO, args, k):
    rewards = []
    agent._actor.eval()

    env = args.env
    device = args.device

    for i in range(args.epochs):
        state = env.reset()
        episode_reward = np.zeros((k))
        t = 0
        while True:
            action = agent.select_action(torch.tensor(state, device=device))
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state
            t += 1
            if done:
                break
        rewards.append(episode_reward)
        print(f"Episode: {i}, length: {t}, ", end='')
        for i in range(episode_reward.shape[0]):
            print(f'reward{i}: {episode_reward[i]} ', end='')
        print()
    avg_reward = np.stack(rewards, axis=-1).mean(axis=-1)
    for i in range(episode_reward.shape[0]):
        print(f'reward{i}: {avg_reward[i]} ', end='')
    print()


def main():
    # TODO: change GaussianMOMPO to Categorical MOMPO
    args = parse_args()
    args.logdir = os.path.join(args.logdir, args.env)
    os.makedirs(args.logdir, exist_ok=True)

    if args.env == 'DeepSeaTreasure':
        args.env = DeepSeaTreasure()
        k = 2
        state_dim = len(args.env.state_spec)
        action_dim = 1
    else:
        raise NotImplementedError

    args.epsilons = np.array([float(x) for x in args.epsilons.split(',')])

    # agent = GaussianMOMPO(state_dim, action_dim, beta=args.beta, gamma=args.gamma, epsilon=args.epsilons)
    agent = GaussianMOMPO(state_dim, action_dim, gamma=args.gamma, epsilon=args.epsilons)
    agent._actor.share_memory()

    if args.model != '':
        agent.load(args.modeldir)

    if args.test_only:
        test(agent, args, k)
    elif args.multiprocess > 1:
        ps = []
        for i in range(args.multiprocess):
            ps.append(mp.Process(target=MultiTrain, args=(agent._actor, args, k, state_dim, action_dim)))
        for p in ps:
            p.start()
        time.sleep(100000)
        Learner(agent, ps)
        for p in ps:
            p.join()
    else:
        SingleTrain(agent, args, k)


if __name__ == '__main__':
    main()

