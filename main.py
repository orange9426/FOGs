import env as env_module
import solver as solver_module
from util import logger
from run import run

import argparse


def parse_args():
    """Parse the arguments and cast to a dictionary."""
    parser = argparse.ArgumentParser(description='Set the run parameters.')

    # Argments for model
    parser.add_argument('--env', default='Tiger', type=str,
                        help='Specify the env to solve {Tiger}')
    parser.add_argument('--solver', default='MEPOP', type=str,
                        help='Specify the solver to use {POMCP}')
    parser.add_argument('--discount', default=1, type=float,
                        help='Specify the discount factor (default=1)')
    parser.add_argument('--n_epochs', default=1000, type=int,
                        help='Num of epochs of the experiment to conduct')
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                        help='Flag of whether to print step messages')

    # Arguments for POMCP
    parser.add_argument('--n_sims', default=100, type=int,
                        help='For POMCP, this is the num of MC sims to do at each belief node')
    parser.add_argument('--n_start_states', default=200, type=int,
                        help='Num of state particles to generate for root belief node in MCTS')
    parser.add_argument('--min_particle_count', default=100, type=int,
                        help='Lower bound on num of particles a belief node can have in MCTS')
    parser.add_argument('--max_particle_count', default=200, type=int,
                        help='Upper bound on num of particles a belief node can have in MCTS')
    parser.add_argument('--max_depth', default=100, type=int,
                        help='Max depth for a DFS of the belief search tree in MCTS')
    parser.add_argument('--uct_coefficient', default=100.0, type=float,
                        help='Coefficient for UCT algorithm used by MCTS')

    # Arguments for MEPOP
    parser.add_argument('--me_tau', default=50.0, type=float,
                        help='Tau for Maximum Entropy algorithm used by MCTS')
    parser.add_argument('--me_epsilon', default=0.0, type=float,
                        help='Epsilon for Maximum Entropy algorithm used by MCTS')

    parser.set_defaults(quiet=True)

    # Cast to a dictionary
    args = vars(parser.parse_args())

    return args


def main():
    # Parse the arguments for each run
    args = parse_args()

    # Init the logger
    logger.init_logger(args['env'], args['solver'])

    # Init the environment
    env = getattr(env_module, args['env'])()

    # Init the solver
    solver = getattr(solver_module, args['solver'])(env, args)

    # Run the experiment
    run(solver, args)


if __name__ == '__main__':
    main()
