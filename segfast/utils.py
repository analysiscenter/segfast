""" !!. """
from functools import partial
from concurrent.futures import Future, Executor
import warnings
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


class TraceHeader:
    """ Trace header class to store its name and byte position. By default, byte position is defined by name
    accordingly to SEG-Y specification.

    Parameters
    ----------
    name : str or int
        If str, name of the header. If int, is interpreted as ``'byte'`` and name will be default from spec.
    byte : int, optional
        Byte position of the header, by default ``None``. If ``None``, default byte position from the spec will be used.
    dtype : int ot str, optional
        dtype for header (e.g. ``'i2'``, ``'f4'``) or its length in bytes (then is interpreted as integer type).
    """
    TRACE_HEADER_SIZE = 240

    def __init__(self, name, byte=None, dtype=None):
        standard_byte_to_header = {v: k for k, v in segyio.tracefield.keys.items()}
        start_bytes = sorted(segyio.tracefield.keys.values())
        standard_byte_to_len = {start: end - start
                                for start, end in zip(start_bytes, start_bytes[1:] + [self.TRACE_HEADER_SIZE + 1])}

        if isinstance(name, int):
            if byte is not None:
                raise ValueError("'name' is int and 'byte' is defined")
            name = standard_byte_to_header[name]

        if name not in segyio.tracefield.keys:
            warnings.warn(f'{name} is not a standard header name')

        self.name = name
        self.byte = byte or segyio.tracefield.keys[name]
        self.standard_name = standard_byte_to_header.get(self.byte)
        self.dtype = dtype or standard_byte_to_len[self.byte]
        self.byte_len = self.dtype

        if isinstance(self.dtype, int):
            self.dtype = 'i' + str(self.dtype)

        if isinstance(self.byte_len, str):
            self.byte_len = int(self.byte_len[-1])

        if self.byte + self.byte_len > self.TRACE_HEADER_SIZE:
            raise ValueError(f'{self.name} header position is out of bounds')

    def __eq__(self, other):
        """ Comparison of two headers by its attributes. """
        if isinstance(other, str):
            return self.name == other

        return self.__dict__ == other.__dict__



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
