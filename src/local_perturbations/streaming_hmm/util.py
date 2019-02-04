import sys


def diff(first, second):
    """Takes the set diff of two lists"""
    second = set(second)
    return [item for item in first if item not in second]


def print_with_carriage_return(msg):
    """
    :param msg: String.
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
