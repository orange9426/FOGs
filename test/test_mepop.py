import sys
sys.path.append(sys.path[0] + '\..')

import env as env_module
import solver as solver_module
from util import logger
from run import run

import numpy as np
from matplotlib import pyplot as plt


def main():
    # Parse the arguments for each run
    args = {'env': 'Tiger',
            'solver': 'MEPOP',
            'discount': 1,
            'n_epochs': 1000,
            'quiet': True,
            'n_sims': 100,
            'n_start_states': 200,
            'min_particle_count': 100,
            'max_particle_count': 200,
            'max_depth': 100,
            'uct_coefficient': 100,
            'me_tau': 0.4,
            'me_epsilon': 0.5
            }

    env = getattr(env_module, args['env'])()

    # # Test the arguments of POMCP
    # args['solver'] = 'POMCP'
    # logger.init_logger(args['env'], args['solver'])

    # for uct_c in range(0, 160, 10):
    #     args['uct_coefficient'] = uct_c

    #     solver = getattr(solver_module, args['solver'])(env, args)
    #     run(solver, args)

    # # Test the arguments of MEPOP
    # args['solver'] = 'MEPOP'
    # logger.init_logger(args['env'], args['solver'])

    # for me_tau in np.arange(0.2, 2.2, 0.2):
    #     for me_epsilon in [0.75]:
    #         args['me_tau'] = me_tau
    #         args['me_epsilon'] = me_epsilon

    #         solver = getattr(solver_module, args['solver'])(env, args)
    #         run(solver, args)

    # Test the POMCP
    args['solver'] = 'POMCP'
    logger.init_logger(args['env'], args['solver'])
    pomcp_history = []

    for n_sims in range(50, 1050, 50):
        args['n_sims'] = n_sims

        solver = getattr(solver_module, args['solver'])(env, args)
        result = run(solver, args)
        pomcp_history.append(result.undiscounted_return.mean)

    # Test the MEPOP
    args['solver'] = 'MEPOP'
    logger.init_logger(args['env'], args['solver'])
    mepop_history = []

    for n_sims in range(50, 1050, 50):
        args['n_sims'] = n_sims

        solver = getattr(solver_module, args['solver'])(env, args)
        result = run(solver, args)
        mepop_history.append(result.undiscounted_return.mean)
    
    X = list(range(50, 150, 50))
    plt.figure()
    plt.plot(X, pomcp_history, label='POMCP')
    plt.plot(X, mepop_history, label='MEPOP')
    plt.title('Algorithms in Tiger')
    plt.xlabel('num of simulations')
    plt.ylabel('total return')
    plt.legend()
    plt.savefig('results/graph/mepop.png')


if __name__ == '__main__':
    main()
