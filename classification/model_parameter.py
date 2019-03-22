import sys


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def get_parameter_526_26():
    h = 526
    w = 26
    kernel_size = (5, 5)
    kernel_stride = (1, 1)
    pool_stride = (4, 1)
    pool_size_list = [(4, 1), (4, 1), (2, 1), (2, 2)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_149_26():
    h = 149
    w = 26
    kernel_size = (5, 5)
    kernel_stride = (1, 1)
    pool_stride = (2, 1)
    pool_size_list = [(2, 1), (2, 1), (2, 1), (2, 2)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_99_26():
    h = 99
    w = 26
    kernel_size = (5, 5)
    kernel_stride = (1, 1)
    pool_stride = (2, 1)
    pool_size_list = [(2, 1), (2, 1), (2, 1), (2, 2)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_49_26():
    h = 49
    w = 26
    kernel_size = (3, 3)
    kernel_stride = (1, 1)
    pool_stride = (2, 1)
    pool_size_list = [(2, 1), (2, 1), (2, 1), (2, 2)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list
