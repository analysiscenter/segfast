""" A thin wrapper around `segyio` for convenient loading of seismic traces. """
from copy import copy
import warnings

import numpy as np
import pandas as pd

import segyio
from .trace_header_spec import TraceHeaderSpec
from .utils import Notifier



class SegyioLoader:
    """ A thin wrapper around **segyio** library for convenient loading of headers and traces.

    Most of the methods directly call the public API of **segyio**.

    For trace loading, we use private methods and attributes of :class:`segyio.SegyFile`, which allow:

       * reading data into the pre-defined buffer
       * read only parts of the trace

    This gives up to 50% speed-up over public API for the scenario of loading sequence of traces,
    and up to 15% over public API in case of loading full lines (inlines or crosslines).
    """
    SEGY_FORMAT_TO_TRACE_DATA_DTYPE = {
        1:  "u1",  # IBM 4-byte float: has to be manually transformed to an IEEE float32
        2:  "i4",
        3:  "i2",
        5:  "f4",
        6:  "f8",
        8:  "i1",
        9:  "i8",
        10: "u4",
        11: "u2",
        12: "u8",
        16: "u1",
    } #: :meta private:

    ENDIANNESS_TO_SYMBOL = {
        "big": ">",
        "msb": ">",

        "little": "<",
        "lsb": "<",
    } #: :meta private:

    def __init__(self, path, endian='big', strict=False, ignore_geometry=True):
        # Parse arguments for errors
        if endian not in self.ENDIANNESS_TO_SYMBOL:
            raise ValueError(f'Unknown endian {endian}, must be one of {self.ENDIANNESS_TO_SYMBOL}')

        # Store arguments
        self.path = path
        self.endian = endian
        self.strict = strict
        self.ignore_geometry = ignore_geometry

        # Open SEG-Y file
        self.file_handler = segyio.open(path, mode='r', endian=endian,
                                        strict=strict, ignore_geometry=ignore_geometry)
        self.file_handler.mmap()

        # Number of traces and depth
        self.n_samples = self.file_handler.trace.shape
        self.n_traces = self.file_handler.trace.length
        self.dtype = self.file_handler.dtype

        # Misc
        self.metrics = self.file_handler.xfd.metrics()
        self.text = [self.file_handler.text[i] for i in range(1 + self.file_handler.ext_headers)]

    @property
    def sample_interval(self):
        """ Sample interval of seismic traces. """
        bin_sample_interval = self.file_handler.bin[segyio.BinField.Interval]
        trace_sample_interval = self.file_handler.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
        # 0 means undefined sample interval, so it is removed from the set
        union_sample_interval = {bin_sample_interval, trace_sample_interval} - {0}

        if len(union_sample_interval) != 1:
            raise ValueError("Cannot infer sample interval from file headers: "
                             "either both `Interval` (bytes 3217-3218 in the binary header) "
                             "and `TRACE_SAMPLE_INTERVAL` (bytes 117-118 in the header of the first trace) "
                             "are undefined or they have different values.")
        return union_sample_interval.pop()

    @property
    def delay(self):
        """ Delay recording time of seismic traces. """
        return self.file_handler.header[0].get(segyio.TraceField.DelayRecordingTime)


    # Headers
    def load_headers(self, headers, indices=None, reconstruct_tsf=True, sort_columns=True, return_specs=False,
                     tracewise=True, pbar=False, **kwargs):
        """ Load requested trace headers from a SEG-Y file for each trace into a dataframe.
        If needed, we reconstruct the ``'TRACE_SEQUENCE_FILE'`` manually be re-indexing traces.

        Parameters
        ----------
        headers : sequence
            An array-like where each element can be:
                - ``str`` -- header name,
                - ``int`` -- header starting byte,
                - :class:`~.trace_header_spec.TraceHeaderSpec` -- used as is,
                - ``tuple`` -- args to init :class:`~.trace_header_spec.TraceHeaderSpec`,
                - ``dict`` -- kwargs to init :class:`~.trace_header_spec.TraceHeaderSpec`.

            Note that for :class:`.SegyioLoader` all nonstandard headers byte positions and dtypes will be ignored.
        indices : sequence or None
            Indices of traces to load trace headers for. If not given, trace headers are loaded for all traces.
        reconstruct_tsf : bool
            Whether to reconstruct ``TRACE_SEQUENCE_FILE`` manually.
        sort_columns : bool
            Whether to sort columns in the resulting dataframe by their starting bytes.
        return_specs : bool
            Whether to return header specs used to load trace headers.
        tracewise : bool
            Whether to iterate over the file in a trace-wise manner, instead of header-wise.
        pbar : bool, str
            If ``bool``, then whether to display the progress bar over the file sweep.
            If ``str``, then type of progress bar to display: ``'t'`` for textual, ``'n'`` for widget.

        Return
        ------
        ``pandas.DataFrame``
        """
        _ = kwargs
        headers = self.make_headers_specs(headers)
        if not all(header.has_standard_location for header in headers):
            warnings.warn("All nonstandard trace headers byte positions and dtypes will be ignored.")

        if reconstruct_tsf:
            headers = [header for header in headers if header.name != 'TRACE_SEQUENCE_FILE']

        # Load data to buffer
        n_traces = self.n_traces if indices is None else len(indices)
        buffer = np.empty((n_traces, len(headers)), dtype=np.int32)

        if tracewise:
            indices_iter = range(n_traces) if indices is None else indices
            for i, trace_ix in Notifier(pbar, total=n_traces, frequency=1000)(enumerate(indices_iter)):
                trace_headers = self.file_handler.header[trace_ix]
                for j, header in enumerate(headers):
                    buffer[i, j] = trace_headers.getfield(trace_headers.buf, header.start_byte)
        else:
            indexer = slice(None) if indices is None else indices
            for i, header in enumerate(headers):
                buffer[:, i] = self.file_handler.attributes(header.start_byte)[indexer]

        # Convert to pd.DataFrame, optionally add TSF and sort
        dataframe = pd.DataFrame(buffer, columns=[item.name for item in headers], copy=False)
        dataframe, headers = self.postprocess_headers_dataframe(dataframe, headers=headers, indices=indices,
                                                                reconstruct_tsf=reconstruct_tsf,
                                                                sort_columns=sort_columns)
        if return_specs:
            return dataframe, headers
        return dataframe

    def load_header(self, header, indices=None, **kwargs):
        """ Load exactly one header. """
        return self.load_headers([header], indices=indices, reconstruct_tsf=False, sort_columns=False, **kwargs)

    @staticmethod
    def postprocess_headers_dataframe(dataframe, headers, indices=None, reconstruct_tsf=True, sort_columns=True):
        """ Optionally add ``'TRACE_SEQUENCE_FILE'`` header and sort columns of a headers dataframe. 

        :meta private:
        """
        if reconstruct_tsf:
            if indices is None:
                dtype = np.int32 if len(dataframe) < np.iinfo(np.int32).max else np.int64
                tsf = np.arange(1, len(dataframe) + 1, dtype=dtype)
            else:
                tsf = np.array(indices) + 1
            dataframe['TRACE_SEQUENCE_FILE'] = tsf
            headers = headers + [TraceHeaderSpec('TRACE_SEQUENCE_FILE')]

        if sort_columns:
            headers_indices = np.argsort([item.start_byte for item in headers])
            headers = [headers[i] for i in headers_indices]
            dataframe = dataframe[[header.name for header in headers]]
        return dataframe, headers

    def make_headers_specs(self, headers):
        """ Transform headers list to list of :class:`~.trace_header_spec.TraceHeaderSpec` instances. """
        byteorder = self.ENDIANNESS_TO_SYMBOL[self.endian]

        if headers == 'all':
            return [TraceHeaderSpec(start_byte=start_byte, byteorder=byteorder)
                    for start_byte in TraceHeaderSpec.STANDARD_BYTE_TO_HEADER]

        headers_ = []
        for header in headers:
            if isinstance(header, TraceHeaderSpec):
                headers_.append(header.set_default_byteorder(byteorder))
            else:
                if isinstance(header, int):
                    header = (None, header)
                if isinstance(header, str):
                    header = (header,)
                if isinstance(header, dict):
                    init_kwargs = header
                else:
                    init_kwargs = dict(zip(['name', 'start_byte', 'dtype'], header))
                init_kwargs = {'byteorder': byteorder, **init_kwargs}
                headers_.append(TraceHeaderSpec(**init_kwargs))
        return headers_


    # Data loading: traces
    def load_traces(self, indices, limits=None, buffer=None):
        """ Load traces by their indices.
        By pre-allocating memory for all of the requested traces, we significantly speed up the process.

        Parameters
        ----------
        indices : sequence
            Indices (``TRACE_SEQUENCE_FILE``) of the traces to read.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth axis.
        buffer : numpy.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        limits = self.process_limits(limits)
        n_samples = len(range(*limits.indices(self.n_samples)))

        if buffer is None:
            buffer = np.empty((len(indices), n_samples), dtype=self.dtype)

        for i, index in enumerate(indices):
            self.load_trace(index=index, buffer=buffer[i], limits=limits)
        return buffer

    def process_limits(self, limits):
        """ Convert given ``limits`` to a ``slice`` instance. """
        if limits is None:
            return slice(0, self.n_samples, 1)
        if isinstance(limits, int):
            limits = slice(limits)
        elif isinstance(limits, (tuple, list)):
            limits = slice(*limits)

        # Use .indices to avoid negative slicing range
        indices = limits.indices(self.n_samples)
        if indices[-1] < 0:
            raise ValueError('Negative step is not allowed.')
        if indices[1] <= indices[0]:
            raise ValueError('Empty traces after setting limits.')
        return slice(*indices)

    def load_trace(self, index, buffer, limits):
        """ Load one trace into the buffer. """
        self.file_handler.xfd.gettr(buffer, index, 1, 1,
                                    limits.start, limits.stop, limits.step,
                                    buffer.size)


    # Data loading: depth slices
    def load_depth_slices(self, indices, buffer=None):
        """ Load horizontal (depth) slices of the data.
        Requires an almost full sweep through SEG-Y, therefore is slow.

        Parameters
        ----------
        indices : sequence
            Indices (ordinals) of the depth slices to read.
        buffer : numpy.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.

        Return
        ------
        numpy.ndarray
        """
        if buffer is None:
            buffer = np.empty((len(indices), self.n_traces), dtype=self.dtype)

        for i, index in enumerate(indices):
            self.load_depth_slice(index=index, buffer=buffer[i])
        return buffer

    def load_depth_slice(self, index, buffer):
        """ Load one depth slice into buffer. """
        self.file_handler.xfd.getdepth(index, buffer.size, 1, buffer)


    # Convenience and utility methods
    def make_chunk_iterator(self, chunk_size=None, n_chunks=None, limits=None, buffer=None):
        """ Create an iterator over the entire file traces in chunks.

        Each chunk contains no more than ``chunk_size`` traces.
        If ``chunk_size`` is not provided and ``n_chunks`` is given instead, there are no more than ``n_chunks`` chunks.
        One and only one of ``chunk_size`` and ``n_chunks`` should be provided.

        Each element in the iterator is a dictionary with ``'data'``, ``'start'`` and ``'end'`` keys.

        Parameters
        ----------
        chunk_size : int, optional
            Maximum size of the chunk.
        n_chunks : int, optional
            Maximum number of chunks.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth (last) axis. Passed directly to :meth:`.load_traces`.
        buffer : numpy.ndarray, optional
            Buffer to read the data into. If possible, avoids copies. Passed directly to :meth:`.load_traces`.

        Return
        ------
        iterator, info : tuple with two elements

        iterator : iterable
            An iterator over the entire SEG-Y traces.
            Each element in the iterator is a dictionary with ``'data'``, ``'start'`` and ``'end'`` keys.
        info : dict
            Description of the iterator with ``'chunk_size'``, ``'n_chunks'``, ``'chunk_starts'`` and ``'chunk_ends'``
            keys.
        """
        # Parse input parameters
        if chunk_size is None and n_chunks is None:
            raise ValueError('Either `chunk_size` or `n_chunks` should be provided!')
        if chunk_size is not None and n_chunks is not None:
            raise ValueError('Only one of `chunk_size` and `n_chunks` should be provided!')

        if n_chunks is not None:
            chunk_size = self.n_traces // n_chunks

        # Define start and end for each chunk
        n_chunks, last_chunk_size = divmod(self.n_traces, chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            n_chunks += 1
            chunk_sizes += [last_chunk_size]

        chunk_starts = np.cumsum([0] + chunk_sizes[:-1])
        chunk_ends = np.cumsum(chunk_sizes)

        # Prepare iterator
        iterator = ({'data': self.load_traces(list(range(start, end)), limits=limits, buffer=buffer),
                     'start': start, 'end': end} for start, end in zip(chunk_starts, chunk_ends))
        info = {
            'chunk_size': chunk_size,
            'n_chunks': n_chunks,
            'chunk_starts': chunk_starts,
            'chunk_ends': chunk_ends
        }
        return iterator, info

    def chunk_iterator(self, chunk_size=None, n_chunks=None, limits=None, buffer=None):
        """ A shorthand for :meth:`.make_chunk_iterator` with no info returned. """
        return self.make_chunk_iterator(chunk_size=chunk_size, n_chunks=n_chunks,
                                        limits=limits, buffer=buffer)[0]


    # Inner workings
    def __enter__(self):
        return self

    def __exit__(self, _, __, ___):
        self.file_handler.close()

    def __getstate__(self):
        """ Create a pickling state from ``__dict__`` by setting SEG-Y file handler to ``None``. """
        state = copy(self.__dict__)
        state["file_handler"] = None
        return state

    def __setstate__(self, state):
        """ Recreate instance from unpickled state and reopen source SEG-Y file. """
        self.__dict__ = state
        self.file_handler = segyio.open(self.path, mode='r', endian=self.endian,
                                        strict=self.strict, ignore_geometry=self.ignore_geometry)
        self.file_handler.mmap()


class SafeSegyioLoader(SegyioLoader):
    """ A thin wrapper around **segyio** library for convenient loading of headers and traces.

    Unlike :class:`.SegyioLoader`, uses only public APIs to load traces.

    Used mainly for performance measurements.
    """
    def load_trace(self, index, buffer, limits):
        """ Load one trace into buffer. """
        buffer[:] = self.file_handler.trace.raw[index][limits]

    def load_depth_slice(self, index, buffer):
        """ Load one depth slice into buffer. """
        buffer[:] = self.file_handler.depth_slice[index]
