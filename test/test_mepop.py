import sys
sys.path.append(sys.path[0] + '\..')

import env as env_module
import solver as solver_module
from util import logger
from run import run

import numpy as np


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
            'uct_coefficient': 80,
            'me_tau': 1,
            'me_epsilon': 0.5
            }

    env = getattr(env_module, args['env'])()

    # Test the arguments of POMCP
    args['solver'] = 'POMCP'
    logger.init_logger(args['env'], args['solver'])

    for uct_c in range(0, 160, 10):
        args['uct_coefficient'] = uct_c

        solver = getattr(solver_module, args['solver'])(env, args)
        run(solver, args)

    # Test the arguments of MEPOP
    args['solver'] = 'MEPOP'
    logger.init_logger(args['env'], args['solver'])

    for me_tau in np.arange(0, 20.5, 0.5):
        for me_epsilon in np.arange(0, 1.1, 0.1):

            solver = getattr(solver_module, args['solver'])(env, args)
            run(solver, args)


if __name__ == '__main__':
    main()
