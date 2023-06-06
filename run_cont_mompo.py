from agents.mpo_humanoid import GaussianMOMPOHumanoid, BehaviorGaussianMPO
from dm_control import viewer
from envs.humanoid.MO_humannoid import MOHumanoid_run

import argparse
import os
from collections import namedtuple
import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--logdir', default='./logs', type=str, help='base directory to save log')
    parse.add_argument('--model', default='', type=str, help='pretrained model path')
    parse.add_argument('--env', default='humanoid_run', type=str, help='environment name')
    parse.add_argument('--beta_mean', default=5e-3, type=int, help='KL constraint on the change of policy mean')
    parse.add_argument('--beta_std', default=1e-3, type=int, help='KL constraint on the change of policy variance')
    parse.add_argument('--gamma', default=0.999, type=int, help='discount factor')
    parse.add_argument('--tanh_on_action_mean', default=False, action='store_true')
    parse.add_argument('--tanh_on_action', default=False, action='store_true')
    parse.add_argument('--min_std', default=1e-4, type=float, help='minimum variance on gaussian of policy network output')
    parse.add_argument('--lr', default=2e-4, type=float, help='learning rate')
    parse.add_argument('--adam_eps', default=1e-8, type=float, help='epsilon of Adam optimizer')
    parse.add_argument('--epsilons', default='0.1,0.000001', type=str, help='epsilon of different objective')
    parse.add_argument('--eps', default=1., type=float, help='epsilon for exploration')
    parse.add_argument('--eps_min', default=0.1, type=int, help='minimum of epsilon for exploration')
    parse.add_argument('--eps_decay', default=10000, type=int, help='minimum of epsilon for exploration')
    parse.add_argument('--test_only', default=False, action='store_true')
    parse.add_argument('--train_iter', default=30000, type=int, help='training iterations')
    parse.add_argument('--test_iter', default=10, type=int, help='testing iterations')
    parse.add_argument('--device', default='cpu', type=str, help='device')
    parse.add_argument('--seed', default=10, type=int, help='random seed')
    parse.add_argument('--multiprocess', default=1, type=int, help='how many process for asynchronous actor')

    args = parse.parse_args()

    return args


def SingleTrain(agent: GaussianMOMPOHumanoid, args, k):
    env = args.env
    device = args.device

    writer = SummaryWriter(args.logdir)
    stop_episode = 30
    stop_episode_reward = np.zeros((k,))
    i = 0
    while True:
        i += 1
        state = env.reset()
        t = 0
        episode_reward = np.zeros((k,))
        trajectory = []
        while True:
            action, log_prob = agent.select_action(torch.tensor(state, dtype=torch.float, device=device))
            next_state, reward, done = env.step(action[0])
            trajectory.append((state, action, reward, log_prob, [int(done)]))
            state = next_state
            episode_reward += reward
            t += 1
            args.eps -= (1 - args.eps_min) / args.eps_decay
            args.eps = max(args.eps, args.eps_min)
            if done:
                break

        agent._replay_buffer.push(trajectory)
    
        agent.update(i)

        # print result
        print(f"Episode: {i}, length: {t} ", end='')
        for j in range(episode_reward.shape[0]):
            print(f'reward{j}: {episode_reward[j]} ', end='')
        print()

        # log result in tensorboard
        for j in range(episode_reward.shape[0]):
            writer.add_scalar(f'reward_{j}', episode_reward[j], i)

        # check convergence
        if np.array_equal(episode_reward, stop_episode_reward):
            print(stop_episode)
            stop_episode -= 1
            if stop_episode == 0:
                with open(os.path.join(args.logdir, 'convergence.txt'), 'w') as f:
                    f.write(f'Epsiode: {i}\n')
                    f.write('Converge at: ')
                    for j in range(episode_reward.shape[0]):
                        f.write(f'reward{j}: {episode_reward[j]} ')
                break
        else:
            stop_episode = 30
            stop_episode_reward = episode_reward

    agent.save(args.logdir)




def MultiTrain(shared_actor, args, k, state_dim, action_dim):
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
            trajectory.append((state, [action], reward, [log_prob], [int(done)]))
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


def Learner(agent: GaussianMOMPOHumanoid, ps):
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


def test(agent: GaussianMOMPOHumanoid, args, k):
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
    args.logdir = os.path.join(args.logdir, args.env, args.epsilons)
    os.makedirs(args.logdir, exist_ok=True)

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.env == 'humanoid_run':
        args.env = MOHumanoid_run()
        # set hyperparameter
        state_dim = 67
        action_dim = 21
        k = 1
        policy_layer_size = (400, 400, 300)
        critic_layer_size = (500, 500, 400)
    else:
        raise NotImplementedError

    # def random_policy(time_step):
    #     del time_step  # Unused.
    #     return np.random.uniform(low=args.env._env.action_spec().minimum,
    #                             high=args.env._env.action_spec().maximum,
    #                             size=args.env._env.action_spec().shape)

    # viewer.launch(args.env._env, policy=random_policy)


    args.epsilons = np.array([float(x) for x in args.epsilons.split(',')])

    agent = GaussianMOMPOHumanoid(state_dim, 
                                  action_dim, 
                                  policy_layer_size=policy_layer_size,
                                  tanh_on_action_mean=args.tanh_on_action_mean,
                                  min_std=args.min_std,
                                  critic_layer_size=critic_layer_size,
                                  tanh_on_action=args.tanh_on_action,
                                  beta_mean=args.beta_mean,
                                  beta_std=args.beta_std,
                                  lr=args.lr,
                                  adam_eps=args.adam_eps)
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

