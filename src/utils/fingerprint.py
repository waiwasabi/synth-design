import numpy as np


class Fingerprint:
    def __init__(self, np_arr):
        self.value = np_arr

    def nbits(self):
        return len(self.value[self.value.astype(bool)])


if __name__ == '__main__':
    pass
