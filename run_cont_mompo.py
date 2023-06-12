from agents import GaussianMOMPOHumanoid, BehaviorGaussianMPO
from dm_control import viewer
from envs.humanoid.MO_humannoid import MOHumanoid_run

import argparse
import os
import numpy as np
import time
import random
from threading import Thread

import torch
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
    parse.add_argument('--eps', default=0., type=float, help='epsilon for exploration')
    parse.add_argument('--eps_min', default=0., type=int, help='minimum of epsilon for exploration')
    parse.add_argument('--eps_decay', default=1e5, type=int, help='minimum of epsilon for exploration')
    parse.add_argument('--test_only', default=False, action='store_true')
    parse.add_argument('--train_iter', default=1e7, type=int, help='training iterations')
    parse.add_argument('--test_iter', default=10, type=int, help='testing iterations')
    parse.add_argument('--device', default='cpu', type=str, help='device')
    parse.add_argument('--seed', default=1, type=int, help='random seed')
    parse.add_argument('--multiprocess', default=1, type=int, help='how many process for asynchronous actor')
    parse.add_argument('--dual_lr', default=1e-4, type=float, help='dual variable learning rate')
    parse.add_argument('--warmup', default=1e2, type=float, help='number of warmup epochs')

    args = parse.parse_args()

    return args


def SingleTrain(agent: GaussianMOMPOHumanoid, args, k):
    env = args.env
    device = args.device

    writer = SummaryWriter(args.logdir)
    i = 0
    while True:
        i += 1
        state = env.reset()
        t = 0
        episode_reward = np.zeros((2,))
        trajectory = []
        while True:
            action, log_prob = agent.select_action(torch.tensor(state, dtype=torch.float, device=device))
            next_state, reward, done = env.step(action)
            energy_penalty = np.array([-np.linalg.norm(action)])
            trajectory.append((state, action, reward, next_state, log_prob, [int(done)]))
            state = next_state
            episode_reward += np.concatenate([reward, energy_penalty])
            t += 1
            args.eps -= (1 - args.eps_min) / args.eps_decay
            args.eps = max(args.eps, args.eps_min)
            if done:
                break

        agent._replay_buffer.push(trajectory)
    
        loss = agent.update(i)
        writer.add_scalar('eps', args.eps, i)
        writer.add_scalar('alpha_mean', agent._alpha_mean, i)
        writer.add_scalar('alpha_std', agent._alpha_std, i)
        writer.add_scalars('temperature', dict(zip(['k1', 'k2'], agent._temperatures.tolist())), i)
        writer.add_scalars('loss', loss, i)

        # print result
        print(f"Episode: {i}, length: {t} ", end='')
        for j in range(episode_reward.shape[0]):
            print(f'reward{j}: {episode_reward[j]:.5f} ', end='')
        print()

        # log result in tensorboard
        if i % 100 == 0:
            avg_reward = test(agent, args, k)
            for j in range(avg_reward.shape[0]):
                writer.add_scalar(f'test_reward_{j}', avg_reward[j], i)


def MultiTrain(args, k, state_dim, action_dim, policy_layer_size, replay_buffer_q, actor_q):
    env = args.env
    device = args.device

    # asyncronous actor
    agent = BehaviorGaussianMPO(state_dim, 
                                action_dim, 
                                policy_layer_size=policy_layer_size,
                                tanh_on_action_mean=args.tanh_on_action_mean,
                                min_std=args.min_std,
                                epsilon=args.epsilons, 
                                k=k,
                                device=args.device)

    agent._actor.train()
    print_freq = 1
    episode_reward = np.zeros((2,))

    for i in range(1, int(args.train_iter) + 1): 
        state = env.reset()
        t = 0
        episode_reward = np.zeros((2,))
        trajectory = []
        while True:
            action, log_prob = agent.select_action(torch.tensor(state, dtype=torch.float, device=device))
            next_state, reward, done = env.step(action)
            energy_penalty = np.array([-np.linalg.norm(action)])
            trajectory.append((state, action, reward, next_state, log_prob, [int(done)]))
            episode_reward += np.concatenate([reward, energy_penalty])
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
                print(f'reward{i}: {episode_reward[i]} ', end='')
            print()

        replay_buffer_q.put_nowait(trajectory)
        try:
            actor = actor_q.get_nowait()
            if actor:
                agent._actor.load_state_dict(actor)
                agent._actor.train()
        except:
            pass


def recieve_transition(agent: GaussianMOMPOHumanoid, replay_buffer_q):
    while True:
        while not replay_buffer_q.empty():
            transitions = replay_buffer_q.get()
            agent._replay_buffer.push(transitions)



def Learner(agent: GaussianMOMPOHumanoid, ps, actor_q, replay_buffer_q, args, k):
    writer = SummaryWriter(args.logdir)
    all_ps_finish = False
    t = 0
    # use threads to recieve transition from actor
    threads = [
        Thread(target=recieve_transition, kwargs={'agent': agent, 'replay_buffer_q': replay_buffer_q})
    ]
    for thread in threads:
        thread.start()

    while not all_ps_finish:
        agent._actor.train()
        t += 1
        while not replay_buffer_q.empty():
            transitions = replay_buffer_q.get()
            agent._replay_buffer.push(transitions)
        # wait for replay buffer has element; TODO write it in other way
        while (not agent._replay_buffer._isfull) and agent._replay_buffer._idx == 0:
            pass
        loss = agent.update(t)
        writer.add_scalar('alpha_mean', agent._alpha_mean, t)
        writer.add_scalar('alpha_std', agent._alpha_mean, t)
        writer.add_scalars('temperature', dict(zip(['k1', 'k2'], agent._temperatures.tolist())), t)
        writer.add_scalars('loss', loss, t)
        all_ps_finish = True
        for p in ps:
            if p.is_alive():
                all_ps_finish = False
                break
        if t % 100 == 0:
            avg_reward = test(agent, args, k)
            for j in range(avg_reward.shape[0]):
                writer.add_scalar(f'test_reward_{j}', avg_reward[j], t)
        for _ in range(args.multiprocess):
            actor_q.put(agent._actor.state_dict())


def test(agent: GaussianMOMPOHumanoid, args, k):
    rewards = []
    agent._actor.eval()

    env = args.env
    device = args.device

    for i in range(args.test_iter):
        state = env.reset()
        episode_reward = np.zeros((2,))
        t = 0
        while True:
            action, _ = agent.select_action(torch.tensor(state, dtype=torch.float, device=device))
            next_state, reward, done = env.step(action)
            energy_penalty = np.array([-np.linalg.norm(action)])
            state = next_state
            episode_reward += np.concatenate([reward, energy_penalty])
            t += 1
            if done:
                break
        rewards.append(episode_reward)
    rewards = np.stack(rewards, axis=-1)
    avg_reward = rewards.mean(axis=-1)
    print("[TEST] ", end='')
    for i in range(episode_reward.shape[0]):
        print(f'reward{i}: {avg_reward[i]:.2f} ', end='')
    print()

    # check if all objective rewards are identical
    if (rewards == rewards[0]).all() and avg_reward[0] >= args.tolerance:
        with open(os.path.join(args.logdir, 'convergence.txt'), 'w') as f:
                    f.write(f'Epsiode: {i}\n')
                    f.write('Converge at: ')
                    for j in range(episode_reward.shape[0]):
                        f.write(f'reward{j}: {episode_reward[j]:.2f} ')
    agent.save(args.logdir)
    return avg_reward


def main():
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
                                  dual_lr=args.dual_lr,
                                  adam_eps=args.adam_eps,
                                  epsilon=args.epsilons, 
                                  k=k,
                                  device=args.device)
    agent._actor.share_memory()

    if args.model != '':
        agent.load(args.modeldir)

    if args.test_only:
        test(agent, args, k)
    elif args.multiprocess > 1:
        torch.multiprocessing.set_start_method('spawn')
        replay_buffer_q = mp.Queue()
        actor_q = mp.Queue()
        ps = []
        for i in range(args.multiprocess):
            ps.append(mp.Process(target=MultiTrain, args=(args, k, state_dim, action_dim, policy_layer_size, replay_buffer_q, actor_q)))
        for p in ps:
            p.start()
        Learner(agent, ps, actor_q, replay_buffer_q, args, k)
        for p in ps:
            p.join()
    else:
        SingleTrain(agent, args, k)


if __name__ == '__main__':
    main()

