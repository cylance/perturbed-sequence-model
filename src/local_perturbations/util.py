import string

import sys


def make_vocab(W):
    if W <= 52:
        vocab = [string.ascii_letters[i] for i in range(W)]
    else:
        vocab = [str(i) for i in range(W)]
    return vocab


def print_with_carriage_return(msg):
    """
    :param msg: String.
    """
    sys.stdout.write(msg)
    sys.stdout.flush()
