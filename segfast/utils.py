""" !!. """
from functools import partial
from concurrent.futures import Future, Executor
import numpy as np
import segyio


try:
    from batchflow import Notifier
except ImportError:
    try:
        from tqdm.auto import tqdm
        def Notifier(pbar, *args, **kwargs):
            """ Progress bar. """
            if pbar:
                return tqdm(*args, **kwargs)
            return lambda iterator: iterator
    except ImportError:
        class Notifier:
            """ Dummy notifier. """
            def __init__(self, *args, **kwargs):
                _ = args, kwargs

            def __call__(self, iterator, *args, **kwargs):
                _ = args, kwargs
                return iterator

            def __enter__(self):
                return self

            def __exit__(self, _, __, ___):
                return None

            def update(self, n=1):
                """ Do nothing on update. """
                _ = n
notifier = Notifier


class TraceHeaderSpec:
    """ Trace header class to store its name and byte position. By default, byte position is defined by name
    accordingly to SEG-Y specification.

    Parameters
    ----------
    name : str or int
        If str, name of the header. If int, is interpreted as 'byte' and name will be default from spec.
    byte : int, optional
        Byte position of the header, by default None. If None, default byte position from the spec will be used.
    dtype : int ot str, optional
        dtype for header (e.g. 'i2', 'f4') or its length in bytes (then is interpreted as integer type).
    """
    TRACE_HEADER_SIZE = 240

    STANDARD_BYTE_TO_HEADER = {v: k for k, v in segyio.tracefield.keys.items()}
    START_BYTES = sorted(segyio.tracefield.keys.values())
    STANDARD_START_BYTE_TO_LEN = {start: end - start
                            for start, end in zip(START_BYTES, START_BYTES[1:] + [TRACE_HEADER_SIZE + 1])}

    def __init__(self, name=None, start_byte=None, dtype=None, byteorder=None):
        self.name = name or self.STANDARD_BYTE_TO_HEADER[start_byte]
        self.start_byte = start_byte or segyio.tracefield.keys[name]

        dtype = dtype or self.STANDARD_START_BYTE_TO_LEN[self.start_byte]
        if isinstance(dtype, int):
            dtype = 'i' + str(dtype)

        self.dtype = np.dtype(dtype)
        if dtype[0] not in {'>', '<'} and byteorder is not None:
            self.dtype = self.dtype.newbyteorder(byteorder)

        self.byte_len = self.dtype.itemsize

        if self.start_byte + self.byte_len > self.TRACE_HEADER_SIZE:
            raise ValueError(f'{self.name} header position is out of bounds')

    @property
    def is_standard(self):
        return self.name in segyio.tracefield.keys and \
               self.start_byte == segyio.tracefield.keys[self.name] and \
               self.byte_len == self.STANDARD_START_BYTE_TO_LEN[self.start_byte] and \
               np.issubdtype(self.dtype, np.integer)

    @property
    def is_standard_except_name(self):
        return self.start_byte == segyio.tracefield.keys[self.standard_name] and \
               self.byte_len == self.STANDARD_START_BYTE_TO_LEN[self.start_byte] and \
               np.issubdtype(self.dtype, np.integer)

    @property
    def standard_name(self):
        return self.STANDARD_BYTE_TO_HEADER[self.start_byte]

    def set_default_byteorder(self, byteorder):
        return type(self)(name=self.name, start_byte=self.start_byte, dtype=self.dtype, byteorder=byteorder)


class ForPoolExecutor(Executor):
    """ A sequential executor of tasks in a for loop.
    Inherits `Executor` interface, so can serve as a drop-in replacement for either
    `ThreadPoolExecutor` or `ProcessPoolExecutor` when threads or processes spawning is undesirable.
    """

    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        self.task_queue = []

    def submit(self, fn, /, *args, **kwargs):
        """Schedule `fn` to be executed with given arguments."""
        future = Future()
        self.task_queue.append((future, partial(fn, *args, **kwargs)))
        return future

    def shutdown(self, wait=True):
        """Signal the executor to finish all scheduled tasks and free its resources."""
        _ = wait
        for future, fn in self.task_queue:
            future.set_result(fn())
        self.task_queue = None
