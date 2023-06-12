from agents.mpo_humanoid import CategoricalMOMPO, BehaviorCategoricalMPO
from envs.deep_sea_treasure import DeepSeaTreasure

import argparse
import os
import numpy as np
import time
import random

from sklearn.metrics import mean_absolute_error

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--logdir', default='./logs', type=str, help='base directory to save log')
    parse.add_argument('--model', default='', type=str, help='pretrained model path')
    parse.add_argument('--env', default='DeepSeaTreasure', type=str, help='environment name')
    parse.add_argument('--alpha', default=0.01, type=float, help='the Lagrangian multiplier for the policy update constraint')
    parse.add_argument('--beta', default=0.001, type=float, help='KL constraint on the change of policy')
    parse.add_argument('--gamma', default=0.999, type=float, help='discount factor')
    parse.add_argument('--epsilons', default='0.075,0.05', type=str, help='epsilon of different objective')
    parse.add_argument('--tolerance', default=0.7, type=float, help='the reward tolerance for convergence')
    parse.add_argument('--eps', default=0., type=float, help='epsilon for exploration')
    parse.add_argument('--eps_min', default=0., type=float, help='minimum of epsilon for exploration')
    parse.add_argument('--eps_decay', default=1e5, type=int, help='minimum of epsilon for exploration')
    parse.add_argument('--test_only', default=False, action='store_true')
    parse.add_argument('--train_iter', default=1e7, type=int, help='training iterations')
    parse.add_argument('--test_iter', default=30, type=int, help='testing iterations')
    parse.add_argument('--device', default='cpu', type=str, help='device')
    parse.add_argument('--seed', default=1, type=int, help='random seed')
    parse.add_argument('--multiprocess', default=1, type=int, help='how many process for asynchronous actor')
    parse.add_argument('--replay_buffer_size', default=1e6, type=int, help='replay buffer size')
    parse.add_argument('--dual_lr', default=1e-4, type=float, help='dual variable learning rate')
    parse.add_argument('--warmup', default=1e2, type=float, help='number of warmup epochs')

    args = parse.parse_args()

    return args


action_repr = ['U', 'D', 'L', 'R']
pareto_front_map = {
    0.7 : -1,
    8.2 : -3,
    11.5: -5,
    14.0: -7,
    15.1: -8,
    16.1: -9,
    19.6: -13,
    20.3: -14,
    22.4: -17,
    23.7: -19,
}

def SingleTrain(agent: CategoricalMOMPO, args, k, verbose=False):
    env = args.env
    device = args.device

    writer = SummaryWriter(args.logdir)
    # total_step = 0
    for i in range(1, int(args.train_iter + 1)):
        agent._actor.train()
        state = env.reset()
        t = 0
        episode_reward = np.zeros((k))
        trajectory = []
        while True:
            if i <= args.warmup:
                action = np.random.randint(0, 4, (1,))
                log_prob = np.zeros((1,))
            else:
                action, log_prob = agent.select_action(torch.tensor(state, dtype=torch.float, device=device), args.eps)
            next_state, reward, done = env.step(action[0])
            trajectory.append((state, action, reward, log_prob, [int(done)]))
            agent._replay_buffer.push(state, action, reward, next_state, log_prob, np.array([int(done)]))
            state = next_state
            episode_reward += reward
            t += 1
            # total_step += 1
            args.eps -= (1 - args.eps_min) / args.eps_decay
            args.eps = max(args.eps, args.eps_min)
            if done:
                break

        loss = agent.update(i)
        writer.add_scalar('eps', args.eps, i)
        writer.add_scalar('alpha', agent._alpha, i)
        writer.add_scalars('temperature', dict(zip(['k1', 'k2'], agent._temperatures.tolist())), i)
        writer.add_scalars('loss', loss, i)

        # print result
        print(f"Episode: {i}, length: {t} ", end='')
        for j in range(episode_reward.shape[0]):
            print(f'reward{j}: {episode_reward[j]:.2f} ', end='')
        print()

        # print the transitions
        if i % 20 == 0 and verbose:
            states = [trans[0].tolist() for trans in trajectory] + [state.tolist()]
            actions = [action_repr[trans[1][0]] for trans in trajectory]
            trans = list(zip(states[:-1], actions, states[1:]))
            print(*trans, sep='\n')
            

        # log result in tensorboard
        if i % 100 == 0:
            avg_reward, pareto_front_err = test(agent, args, k)
            writer.add_scalar("pareto-front error", pareto_front_err, i)
            for j in range(avg_reward.shape[0]):
                writer.add_scalar(f'test_reward_{j}', avg_reward[j], i)


def MultiTrain(args, k, state_dim, action_dim, replay_buffer_q, actor_q):
    env = args.env
    device = args.device

    # asyncronous actor
    agent = BehaviorCategoricalMPO(state_dim, action_dim, device=args.device)
    agent._actor.train()
    print_freq = 100
    episode_reward = np.zeros((k))
    for i in range(1, int(args.train_iter) + 1): 
        replay_buffer = []
        state = env.reset()
        t = 0
        while True:
            action, log_prob = agent.select_action(torch.tensor(state, dtype=torch.float, device=device), args.eps)
            next_state, reward, done = env.step(action[0])
            replay_buffer.append((state, action, reward, next_state, log_prob, np.array([int(done)])))
            state = next_state
            episode_reward += reward
            t += 1
            args.eps -= (1 - args.eps_min) / args.eps_decay
            args.eps = max(args.eps, args.eps_min)
            if done:
                break
        
        if i % print_freq == 0:
            print(f"Episode: {i}, length: {t} ")
            for i in range(episode_reward.shape[0]):
                print(f'reward{i}: {episode_reward[i] / print_freq:.2f} ', end='')
            print()
            episode_reward = np.zeros((k))

        replay_buffer_q.put_nowait(replay_buffer)
        try:
            actor = actor_q.get_nowait()
            if actor:
                agent._actor.load_state_dict(actor)
                agent._actor.train()
        except:
            pass


def Learner(agent: CategoricalMOMPO, ps, actor_q, replay_buffer_q, args, k):
    writer = SummaryWriter(args.logdir)
    all_ps_finish = False
    t = 0
    time.sleep(3)
    while not all_ps_finish:
        agent._actor.train()
        t += 1
        while not replay_buffer_q.empty():
            transitions = replay_buffer_q.get()
            for transition in transitions:
                agent._replay_buffer.push(*transition)
        loss = agent.update(t)
        writer.add_scalar('alpha', agent._alpha, t)
        writer.add_scalars('temperature', dict(zip(['k1', 'k2'], agent._temperatures.tolist())), t)
        writer.add_scalars('loss', loss, t)
        all_ps_finish = True
        for p in ps:
            if p.is_alive():
                all_ps_finish = False
                break
        if t % 100 == 0:
            avg_reward, pareto_front_err = test(agent, args, k)
            writer.add_scalar("pareto-front error", pareto_front_err, t)
            for j in range(avg_reward.shape[0]):
                writer.add_scalar(f'test_reward_{j}', avg_reward[j], t)
        for _ in range(args.multiprocess):
            actor_q.put(agent._actor.state_dict())


def test(agent: CategoricalMOMPO, args, k):
    rewards = []
    agent._actor.eval()

    env = args.env
    device = args.device

    for i in range(args.test_iter):
        state = env.reset()
        episode_reward = np.zeros((k))
        t = 0
        while True:
            action, _ = agent.select_action(torch.tensor(state, dtype=torch.float, device=device), 0)
            next_state, reward, done = env.step(action[0])
            episode_reward += reward
            state = next_state
            t += 1
            if done:
                break
        rewards.append(episode_reward)

    # check if pareto-front policy
    rewards0, rewards1 = np.array(list(zip(*rewards)))
    pareto_front = np.array(list(map(pareto_front_map.get, rewards0)))

    pareto_front_err = mean_absolute_error(pareto_front, rewards1)
    is_pareto_front = (rewards1 == pareto_front).all()

    rewards = np.stack(rewards, axis=-1)
    avg_reward = rewards.mean(axis=-1)

    print("[TEST] ", end='')
    for i in range(episode_reward.shape[0]):
        print(f'reward{i}: {avg_reward[i]:.2f} ', end='')

    if is_pareto_front:
        print("(Pareto front)", end='')

    print()

    # check if all objective rewards are identical
    n_converged = np.count_nonzero(rewards[0] >= args.tolerance)
    if is_pareto_front and n_converged > (args.test_iter // 2):
        print(f"Converged with {n_converged} samples satisfying the target reward {args.tolerance}")
        with open(os.path.join(args.logdir, 'convergence.txt'), 'w') as f:
            f.write(f'Epsiode: {i}\n')
            f.write('Converge at: ')
            for j in range(episode_reward.shape[0]):
                f.write(f'reward{j}: {episode_reward[j]:.2f} ')
    
    agent.save(args.logdir)
    return avg_reward, pareto_front_err

def main():
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    args.logdir = os.path.join(args.logdir, args.env, args.epsilons + ',' + str(args.alpha))
    os.makedirs(args.logdir, exist_ok=True)

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.env == 'DeepSeaTreasure':
        args.env = DeepSeaTreasure()
        k = 2
        state_dim = len(args.env.state_spec)
        action_dim = 4
    else:
        raise NotImplementedError

    args.epsilons = np.array([float(x) for x in args.epsilons.split(',')])

    agent = CategoricalMOMPO(state_dim, 
                             action_dim,
                             gamma=args.gamma, 
                             epsilon=args.epsilons, 
                             beta=args.beta, 
                             k=k, 
                             alpha=args.alpha,
                             dual_lr=args.dual_lr,
                             replay_buffer_size=int(args.replay_buffer_size),
                             device=args.device)
    agent._actor.share_memory()

    if args.model != '':
        agent.load(args.model)

    if args.test_only:
        avg_reward, pareto_front_err = test(agent, args, k)
    elif args.multiprocess > 1:
        replay_buffer_q = mp.Queue()
        actor_q = mp.Queue()
        ps = []
        for i in range(args.multiprocess):
            ps.append(mp.Process(target=MultiTrain, args=(args, k, state_dim, action_dim, replay_buffer_q, actor_q)))
        for p in ps:
            p.start()
        Learner(agent, ps, actor_q, replay_buffer_q, args, k)
        for p in ps:
            p.join()
    else:
        SingleTrain(agent, args, k)


if __name__ == '__main__':
    main()

