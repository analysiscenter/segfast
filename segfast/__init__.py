""" Init file. """
from .segyio_loader import SegyioLoader, SafeSegyioLoader
from .memmap_loader import MemmapLoader
from .loader import Loader, File, open #pylint: disable=redefined-builtin

__version__ = '0.9.1'
