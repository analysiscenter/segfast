""" Automatic selector of underlying engine for SEG-Y opening. """
#pylint: disable=redefined-builtin
from .segyio_loader import SegyioLoader
from .memmap_loader import MemmapLoader


def Loader(path, engine='memmap', endian='big', strict=False, ignore_geometry=True):
    """ !!. """
    loader_class = _select_loader_class(engine)
    return loader_class(path=path, endian=endian, strict=strict, ignore_geometry=ignore_geometry)
open = File = Loader


def _select_loader_class(engine):
    if isinstance(engine, type):
        return engine

    engine = engine.lower()
    if engine == 'segyio':
        return SegyioLoader
    if engine == 'memmap':
        return MemmapLoader
    raise ValueError(f'Unknown engine type `{engine}`!')
