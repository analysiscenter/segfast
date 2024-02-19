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
    name : str
        Name of the header.
    start_byte : int, optional
        Byte position of the header, by default None. If None, default byte position from the spec will be used.
    dtype : int, str or dtype, optional
        dtype for header (e.g. 'i2', '>f4', `np.float32`) or its length in bytes (then is interpreted as integer type).
    byteorder : '>' or '<', optional
        Endianness to use, if it's not defined by dtype. If None and dtype doesn't specify, architecture default
        will be used.
    """
    TRACE_HEADER_SIZE = 240

    STANDARD_HEADER_TO_BYTE = segyio.tracefield.keys
    STANDARD_BYTE_TO_HEADER = {v: k for k, v in STANDARD_HEADER_TO_BYTE.items()}

    START_BYTES = sorted(STANDARD_HEADER_TO_BYTE.values())
    STANDARD_BYTE_TO_LEN = {start: end - start
                            for start, end in zip(START_BYTES, START_BYTES[1:] + [TRACE_HEADER_SIZE + 1])}

    def __init__(self, name=None, start_byte=None, dtype=None, byteorder=None):
        self.name = name or self.STANDARD_BYTE_TO_HEADER[start_byte]
        self.start_byte = start_byte or self.STANDARD_HEADER_TO_BYTE[name]

        dtype = dtype or self.STANDARD_BYTE_TO_LEN[self.start_byte]
        if isinstance(dtype, int):
            dtype = 'i' + str(dtype)

        self.dtype = np.dtype(dtype)
        self.default_byteorder = byteorder
        self.has_explicit_byteorder = isinstance(dtype, str) and dtype[0] in {'>', '<'}
        if not self.has_explicit_byteorder and byteorder is not None:
            self.dtype = self.dtype.newbyteorder(byteorder)

        if self.start_byte + self.byte_len > self.TRACE_HEADER_SIZE + 1:
            raise ValueError(f'{self.name} header position is out of bounds')

    @property
    def byte_len(self):
        """ The number of bytes for a header. """
        return self.dtype.itemsize

    @property
    def is_standard(self):
        """ Whether the header matches the specification. """
        return self.name in self.STANDARD_BYTE_TO_HEADER and self.has_standard_location

    @property
    def has_standard_location(self):
        """ Whether the header matches the specification, except maybe the name. """
        return self.start_byte in self.STANDARD_BYTE_TO_HEADER and \
               self.byte_len == self.STANDARD_BYTE_TO_LEN[self.start_byte] and \
               np.issubdtype(self.dtype, np.integer)

    @property
    def standard_name(self):
        """ The name from specification for header (if 'has_standard_location' is True). """
        if not self.has_standard_location:
            raise ValueError("The header has non-standard start byte or dtype")
        return self.STANDARD_BYTE_TO_HEADER[self.start_byte]

    @property
    def has_default_byteorder(self):
        """ Whether default byteorder is defined. """
        return self.default_byteorder is not None

    @property
    def has_byteorder(self):
        """ Whether byteorder is defined. """
        return self.has_explicit_byteorder or self.has_default_byteorder

    @property
    def byteorder(self):
        """ Header byteorder (if defined). """
        if not self.has_byteorder:
            return None
        return self.dtype.str[0]

    @property
    def _spec_params(self):
        dtype_str = self.dtype.str
        if not self.has_byteorder:
            dtype_str = dtype_str[1:]
        return self.name, self.start_byte, dtype_str

    def __eq__(self, other):
        return self._spec_params == other._spec_params

    def __hash__(self):
        return hash(self._spec_params)

    def set_default_byteorder(self, byteorder):
        """ Set byteorder to use as default (if not specified by dtype). """
        dtype = self.dtype.str
        if not self.has_explicit_byteorder:
            dtype = dtype[1:]
        return type(self)(name=self.name, start_byte=self.start_byte, dtype=dtype, byteorder=byteorder)

    def __repr__(self):
        return f"{type(self).__name__}(name='{self.name}', start_byte={self.start_byte}, dtype='{self.dtype.str}')"


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
