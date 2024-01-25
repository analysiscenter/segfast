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
        If str, name of the header. If int, is interpreted as 'byte' and name will be default from spec.
    byte : int, optional
        Byte position of the header, by default None. If None, default byte position from the spec will be used.
    """
    def __init__(self, name, byte=None):
        standard_byte_to_header = {v: k for k, v in segyio.tracefield.keys.items()}

        if isinstance(name, int):
            if byte is not None:
                raise ValueError("'name' is int and 'byte' is defined")
            name = standard_byte_to_header[name]

        if name not in segyio.tracefield.keys:
            warnings.warn(f'{name} is not a standard header name')

        self.name = name
        self.byte = byte or segyio.tracefield.keys[name]
        self.standard_name = standard_byte_to_header[self.byte]

    def __eq__(self, other):
        """ !!. """
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
