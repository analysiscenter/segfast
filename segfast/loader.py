""" Automatic selector of underlying engine for SEG-Y opening. """
#pylint: disable=redefined-builtin
from .segyio_loader import SegyioLoader
from .memmap_loader import MemmapLoader


def Loader(path, engine='segyio', endian='big', strict=False, ignore_geometry=True):
    """ Selector class for loading SEG-Y with either segyio-based loader or memmap-based one.

    Parameters
    ----------
    path : str
        Path to the SEG-Y file
    engine : 'memmap' or 'segyio'
        Engine to load data from file: ``'memmap'`` is based on :class:`numpy.memmap` created for the whole file and
        ``'segyio'`` is for using **segyio** library instruments. in any case, **segyio** is used to load information
        about the entire file (e.g. ``'sample_interval'`` or ``'shape'``).
    endian : 'big' or 'little'
        Byte order in the file.
    strict : bool
        See :func:`segyio.open`
    ignore_geometry : bool
        See :func:`segyio.open`
    Return
    ------
    :class:`~.memmap_loader.MemmapLoader` or :class:`~.segyio_loader.SegyioLoader`
    """
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
