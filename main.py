from log import logger
from env.init_env import init_env
from solver.init_solver import init_solver
from run import run

import argparse
import numpy as np
import time


def parse_args():
    """Parse the arguments and cast to a dictionary."""
    parser = argparse.ArgumentParser(description='Set the run parameters.')

    # Argments for model
    parser.add_argument('--env', default='Kuhn Poker', type=str,
                        help='Specify the env to solve {Tiger}')
    parser.add_argument('--solver', default='POMCP', type=str,
                        help='Specify the solver to use {POMCP}')
    parser.add_argument('--seed', default=int(time.time()), type=int,
                        help='Specify the random seed for numpy.random')
    parser.add_argument('--discount', default=1, type=float,
                        help='Specify the discount factor (default=1)')
    parser.add_argument('--n_epochs', default=10, type=int,
                        help='Num of epochs of the experiment to conduct')
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                        help='Flag of whether to print step messages')

    # Arguments for POMCP
    parser.add_argument('--n_sims', default=100, type=int,
                        help='For POMCP, this is the num of MC sims to do at each belief node')
    parser.add_argument('--n_start_states', default=2000, type=int,
                        help='Num of state particles to generate for root belief node in MCTS')
    parser.add_argument('--min_particle_count', default=1000, type=int,
                        help='Lower bound on num of particles a belief node can have in MCTS')
    parser.add_argument('--max_particle_count', default=2000, type=int,
                        help='Upper bound on num of particles a belief node can have in MCTS')
    parser.add_argument('--max_depth', default=100, type=int,
                        help='Max depth for a DFS of the belief search tree in MCTS')
    parser.add_argument('--uct_coefficient', default=50.0, type=float,
                        help='Coefficient for UCT algorithm used by MCTS')

    # Arguments for ME-POMCP
    parser.add_argument('--me_tau', default=1, type=float,
                        help='Tau for Maximum Entropy algorithm used by MCTS')
    parser.add_argument('--me_epsilon', default=0.5, type=float,
                        help='Epsilon for Maximum Entropy algorithm used by MCTS')

    parser.set_defaults(quiet=False)

    # Cast to a dictionary
    args = vars(parser.parse_args())

    return args


def main():
    # Parse the arguments for each run
    args = parse_args()

    # Init the logger
    logger.init_logger(args['env'], args['solver'])

    # Init the environment
    env = init_env(args['env'])

    # Init the solver
    solver = init_solver(args['solver'], args)

    # Run the process
    run(env, solver, args)


if __name__ == '__main__':
    main()
