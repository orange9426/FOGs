from statistic.results import Results
from util.console import console
from util.divider import print_divider

import logging
import time
import tqdm

module = 'RUN'


def run(env, solver, args):
    """Run multiple epochs as an experiment."""
    # Save all results in the experiment
    results = Results()

    print_divider('large')

    # Run for multiple epochs
    for epoch in tqdm.tqdm(range(args['n_epochs'])):
        # Show epochs progress
        if not args['quiet']:
            print_divider('medium')
            console(2, module, "Epoch: " + str(epoch + 1))

        epoch_start = time.time()

        # Play a game with policies solved by the solver
        game_history = _play_game(env, solver, args)

        # Record the results
        results.time.add(time.time() - epoch_start)
        results.update_reward_results(
            game_history.undiscounted_return(),
            game_history.discounted_return(args['discount']))

        if not args['quiet']:
            print_divider('medium')

    # Show the results
    results.show(args['n_epochs'])

    # Write the results to the log
    _log_result(results, args)


def _play_game(env, solver, args):
    """Plays a game with policies solved by the solver."""
    solver.reset_for_epoch()

    game_history = solver.solve_game(env)

    return game_history


def _log_result(result, args):
    """Write the running result to the log."""
    logger = logging.getLogger(args['env'] + ': ' + args['solver'])

    # Log the results for different solvers
    if args['solver'] == 'POMCP':
        logger.info('epochs: %d' % args['n_epochs'] + '\t' +
                    'simulations: %d' % args['n_sims'] + '\t' +
                    'uct_c: %.3f' % args['uct_coefficient'] + '\t' +
                    'ave undiscounted return: %.3f +- %.3f' %
                    (result.undiscounted_return.mean,
                     result.undiscounted_return.std_err()) + '\t' +
                    'ave discounted return: %.3f +- %.3f' %
                    (result.discounted_return.mean,
                     result.discounted_return.std_err()) + '\t' +
                    'ave time/epoch: %.3f' % result.time.mean)

    elif args['solver'] == 'ME-POMCP':
        logger.info('epochs: %d' % args['n_epochs'] + '\t' +
                    'simulations: %d' % args['n_sims'] + '\t' +
                    'me_tau: %.3f' % args['me_tau'] + '\t' +
                    'me_epsilon: %.3f' % args['me_epsilon'] + '\t' +
                    'ave undiscounted return: %.3f +- %.3f' %
                    (result.undiscounted_return.mean,
                     result.undiscounted_return.std_err()) + '\t' +
                    'ave discounted return: %.3f +- %.3f' %
                    (result.discounted_return.mean,
                     result.discounted_return.std_err()) + '\t' +
                    'ave time/epoch: %.3f' % result.time.mean)
