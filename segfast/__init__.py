""" Init file. """
from .segyio_loader import SegyioLoader, SafeSegyioLoader
from .memmap_loader import MemmapLoader
from .loader import Loader, File, open #pylint: disable=redefined-builtin
from .trace_header_spec import TraceHeaderSpec

__version__ = '1.1.0'
