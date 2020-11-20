from solver.pomcp.pomcp import POMCP
from solver.me_pomcp.me_pomcp import ME_POMCP
from solver.cfr.cfr import CFR


def init_solver(solver_name, args):
    if solver_name == 'POMCP':
        return POMCP(args)
    elif solver_name == 'ME-POMCP':
        return ME_POMCP(args)
    elif solver_name == 'CFR':
        return CFR(args)
    else:
        raise ValueError('Unknown solver: %s' % solver_name)
