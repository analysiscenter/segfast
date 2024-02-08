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

    def __init__(self, name, start_byte=None, dtype=None):
        if isinstance(name, int):
            if start_byte is not None:
                raise ValueError("'name' is int and 'byte' is defined")
            name = self.STANDARD_BYTE_TO_HEADER[name]

        self.name = name
        self.start_byte = start_byte or segyio.tracefield.keys[name]
        self.standard_name = self.STANDARD_BYTE_TO_HEADER.get(self.start_byte)
        self.dtype = dtype or self.STANDARD_START_BYTE_TO_LEN[self.start_byte]
        self.byte_len = self.dtype

        if isinstance(self.dtype, int):
            self.dtype = 'i' + str(self.dtype)

        if isinstance(self.byte_len, str):
            self.byte_len = int(self.byte_len[-1])

        if self.start_byte + self.byte_len > self.TRACE_HEADER_SIZE:
            raise ValueError(f'{self.name} header position is out of bounds')

    @property
    def is_standard(self):
        return self.name in segyio.tracefield.keys and \
               self.start_byte == segyio.tracefield.keys[self.name] and \
               self.byte_len == self.STANDARD_START_BYTE_TO_LEN[self.start_byte]# and \
            #    np.issubdtype(np.dtype, np.integer)



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
