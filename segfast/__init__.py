""" Init file. """
from .segyio_loader import SegyioLoader, SafeSegyioLoader
from .memmap_loader import MemmapLoader
from .loader import Loader, File, open #pylint: disable=redefined-builtin
from .trace_header_spec import TraceHeaderSpec
from .spec_selector import TraceHeaderSpecSelector, select_pre_stack_header_specs, select_post_stack_header_specs

__version__ = '1.0.1'
