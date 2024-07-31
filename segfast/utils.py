""" !!. """
import os
from functools import partial
from importlib import import_module
from concurrent.futures import Future, Executor


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


class DelayedImport:
    """ Proxy class to postpone import until the first access.
    Useful, when the import takes a lot of time (PyTorch, TensorFlow, ahead-of-time compiled functions).

    Note that even access for module inspection (like its documentation or dir) would trigger module loading.

    Examples::
        >>> from .xxx import yyy                                                              # before
        >>> yyy = DelayedImport(module='.xxx', package=__name__, attribute='yyy')             # after

    Parameters
    ----------
    module : str
        Name of module to load.
    package : str, optional
        Anchor for resolving the package name. Used only for relative imports.
    attribute : str, optional
        Name of attribute to get from loaded module.
    help : str, optional
        Additional help on import errors.
    """
    __file__ = globals()["__file__"]
    __path__ = [os.path.dirname(__file__)]

    def __init__(self, module, package=None, attribute=None, help=None):  # pylint: disable=redefined-builtin
        self.module, self.package, self.attribute = module, package, attribute
        self.help = help
        self._loaded_module = None

    @property
    def loaded_module(self):
        """ Try loading the module at the first access. """
        if self._loaded_module is None:
            try:
                self._loaded_module = import_module(self.module, self.package)
                if self.attribute is not None:
                    self._loaded_module = getattr(self._loaded_module, self.attribute)
            except ImportError as e:
                if self.help:
                    raise ImportError(f"No module named '{self.module}'! {self.help}") from e
                raise

        return self._loaded_module

    def __dir__(self):
        return dir(self.loaded_module)

    @property
    def __doc__(self):
        return self.loaded_module.__doc__

    def __getattr__(self, name):
        if name != 'loaded_module':
            return getattr(self.loaded_module, name)
        return super().__getattr__(name)

    def __call__(self, *args, **kwargs):
        # pylint: disable-next=not-callable
        return self.loaded_module(*args, **kwargs)

    def __getitem__(self, key):
        return self.loaded_module[key]
