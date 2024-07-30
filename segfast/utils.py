""" !!. """
from concurrent.futures import Future, Executor
from functools import partial


try:
    from batchflow import Notifier
except ImportError:
    try:
        from tqdm.auto import tqdm

        class Notifier:
            """ tqdm notifier. """
            def __init__(self, bar=False, total=None, **kwargs):
                if 'frequency' in kwargs:
                    kwargs['miniters'] = kwargs.pop('frequency')
                self.pbar = partial(tqdm, disable=not bar, total=total, **kwargs)

            def __call__(self, iterator):
                return self.pbar(iterator)

            def __enter__(self):
                return self.pbar()

            def __exit__(self, _, __, ___):
                pass

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


class ForPoolExecutor(Executor):
    """ A sequential executor of tasks in a for loop.
    Inherits :class:`concurrent.futures.Executor` interface, so can serve as a drop-in replacement for either
    :class:`concurrent.futures.ThreadPoolExecutor` or :class:`concurrent.futures.ProcessPoolExecutor` when threads or
    processes spawning is undesirable.
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
