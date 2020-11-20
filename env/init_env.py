from env.tiger.tiger import Tiger
from env.kuhn_poker.kuhn_poker import KuhnPoker


def init_env(env_name):
    if env_name == 'Tiger':
        return Tiger()
    elif env_name == 'Kuhn Poker':
        return KuhnPoker()
    else:
        raise ValueError('Unknown environment: %s' % env_name)
