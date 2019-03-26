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


def select_parameter(_h, _w):
    # mfcc and logf
    if _w == 26:
        if _h == 526:
            return get_parameter_526_26()
        if _h == 199:
            return get_parameter_199_26()
        if _h == 149:
            return get_parameter_149_26()
        if _h == 99:
            return get_parameter_99_26()
        if _h == 49:
            return get_parameter_49_26()

    # stft
    if _w == 257:
        if _h == 95:
            return get_parameter_95_257()
        if _h == 189:
            return get_parameter_189_257()
        if _h == 283:
            return get_parameter_283_257()
        if _h == 990:
            return get_parameter_990_257()

    return None


def get_parameter_526_26():
    h = 526
    w = 26
    kernel_size = (5, 2)
    kernel_stride = (1, 1)
    pool_stride = (4, 1)
    pool_size_list = [(4, 1), (4, 1), (4, 1), (2, 1)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_199_26():
    h = 199
    w = 26
    kernel_size = (5, 2)
    kernel_stride = (1, 1)
    pool_stride = (2, 1)
    pool_size_list = [(4, 1), (4, 1), (2, 1), (2, 1)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_149_26():
    h = 149
    w = 26
    kernel_size = (5, 2)
    kernel_stride = (1, 1)
    pool_stride = (4, 1)
    pool_size_list = [(4, 1), (2, 1), (2, 1), (2, 1)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_99_26():
    h = 99
    w = 26
    kernel_size = (5, 2)
    kernel_stride = (1, 1)
    pool_stride = (4, 1)
    pool_size_list = [(2, 1), (2, 1), (1, 1), (2, 1)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_49_26():
    h = 49
    w = 26
    kernel_size = (3, 2)
    kernel_stride = (1, 1)
    pool_stride = (4, 1)
    pool_size_list = [(2, 1), (1, 1), (2, 1), (2, 1)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_95_257():
    h = 95
    w = 257
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(1, 2), (1, 2), (1, 2), (2, 4)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_189_257():
    h = 189
    w = 257
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(1, 2), (1, 2), (2, 2), (2, 4)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_283_257():
    h = 283
    w = 257
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(1, 2), (2, 2), (2, 2), (2, 4)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_990_257():
    h = 990
    w = 257
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(2, 2), (2, 2), (4, 2), (4, 4)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list
