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
        if _h == 149:
            return get_parameter_149_26()
        if _h == 99:
            return get_parameter_99_26()
        if _h == 49:
            return get_parameter_49_26()

    # stft
    if _h == 257:
        if _w == 95:
            return get_parameter_257_95()
        if _w == 189:
            return get_parameter_257_189()
        if _w == 283:
            return get_parameter_257_283()
        if _w == 990:
            return get_parameter_257_990()

    return None


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


def get_parameter_257_95():
    h = 257
    w = 95
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(2, 2), (2, 2), (2, 2), (2, 2)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_257_189():
    h = 257
    w = 189
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(2, 2), (2, 2), (2, 2), (2, 2)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_257_283():
    h = 257
    w = 283
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(2, 2), (2, 2), (2, 2), (2, 4)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list


def get_parameter_257_990():
    h = 257
    w = 990
    kernel_size = (5, 5)
    kernel_stride = (2, 2)
    pool_stride = (2, 1)
    pool_size_list = [(2, 2), (2, 4), (2, 4), (2, 4)]
    return h, w, kernel_size, kernel_stride, pool_stride, pool_size_list
