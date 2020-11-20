"""
CONSOLE LOGGING VERBOSITY LEVELS
---------------------------------
0 - FATAL
1 - CRITICAL
2 - INFO
3 - LOUD
4 - DEBUG
"""

VERBOSITY = 4


def console(verbosity_level, module, msg):
    if verbosity_level > VERBOSITY:
        return
    else:
        print(module + ' - ' + msg)
