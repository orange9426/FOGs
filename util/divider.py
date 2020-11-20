"""
Print the dividing line of three sizes.
"""


def print_divider(size="medium"):
    if size == "large":
        print("======================================================================")
    elif size == "medium":
        print("==========================================")
    elif size == "small":
        print("==============")