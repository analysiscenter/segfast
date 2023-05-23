""" A thin wrapper around `segyio` for convenient loading of seismic traces. """
from copy import copy

import numpy as np
import pandas as pd

import segyio
from .utils import Notifier



class SegyioLoader:
    """ A thin wrapper around `segyio` library for convenient loading of headers and traces.

    Most of the methods directly call public API of `segyio`.
    For trace loading we use private methods and attributes of `segyio.SegyFile`, which allow:
        - reading data into pre-defined buffer
        - read only parts of the trace.
    This gives up to 50% speed-up over public API for the scenario of loading sequence of traces,
    and up to 15% over public API in case of loading full lines (inlines or crosslines).
    """
    TRACE_HEADER_SIZE = 240

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
    }

    ENDIANNESS_TO_SYMBOL = {
        "big": ">",
        "msb": ">",

        "little": "<",
        "lsb": "<",
    }

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
    def headers_to_bytes(self, headers):
        """ Compute the byte location of a header. """
        return [getattr(segyio.TraceField, header) for header in headers]

    def postprocess_headers_dataframe(self, dataframe, headers, reconstruct_tsf=True, sort_columns=True):
        """ Optionally add TSF header and sort columns of a headers dataframe. """
        if reconstruct_tsf:
            dataframe['TRACE_SEQUENCE_FILE'] = self.make_tsf_header()
            headers.append('TRACE_SEQUENCE_FILE')

        if sort_columns:
            headers_bytes = self.headers_to_bytes(headers)
            columns = np.array(headers)[np.argsort(headers_bytes)]
            dataframe = dataframe[columns]
        return dataframe

    def load_headers(self, headers, reconstruct_tsf=True, sort_columns=True, tracewise=True, pbar=False, **kwargs):
        """ Load requested trace headers from a SEG-Y file for each trace into a dataframe.
        If needed, we reconstruct the `'TRACE_SEQUENCE_FILE'` manually be re-indexing traces.

        Parameters
        ----------
        headers : sequence
            Names of headers to load.
        reconstruct_tsf : bool
            Whether to reconstruct `TRACE_SEQUENCE_FILE` manually.
        sort_columns : bool
            Whether to sort columns in the resulting dataframe by their starting bytes.
        tracewise : bool
            Whether to iterate over the file in a trace-wise manner, instead of header-wise.
        pbar : bool, str
            If bool, then whether to display progress bar over the file sweep.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        """
        _ = kwargs
        headers = list(headers)

        if reconstruct_tsf and 'TRACE_SEQUENCE_FILE' in headers:
            headers.remove('TRACE_SEQUENCE_FILE')

        headers_bytes = self.headers_to_bytes(headers)

        # Load data to buffer
        buffer = np.empty((self.n_traces, len(headers)), dtype=np.int32)
        if tracewise:
            for i, header_ in Notifier(pbar, total=self.n_traces, frequency=1000)(enumerate(self.file_handler.header)):
                for j, byte_ in enumerate(headers_bytes):
                    buffer[i, j] = header_.getfield(header_.buf, byte_)
        else:
            for i, header in enumerate(headers):
                buffer[:, i] = self.load_header(header)

        # Convert to pd.DataFrame, optionally add TSF and sort
        dataframe = pd.DataFrame(buffer, columns=headers, copy=False)
        dataframe = self.postprocess_headers_dataframe(dataframe, headers=headers,
                                                       reconstruct_tsf=reconstruct_tsf, sort_columns=sort_columns)
        return dataframe

    def load_header(self, header):
        """ Read one header from the file. """
        return self.file_handler.attributes(getattr(segyio.TraceField, header))[:]

    def make_tsf_header(self):
        """ Reconstruct the `TRACE_SEQUENCE_FILE` header. """
        dtype = np.int32 if self.n_traces < np.iinfo(np.int32).max else np.int64
        return np.arange(1, self.n_traces + 1, dtype=dtype)


    # Data loading: traces
    def load_traces(self, indices, limits=None, buffer=None):
        """ Load traces by their indices.
        By pre-allocating memory for all of the requested traces, we significantly speed up the process.

        Parameters
        ----------
        indices : sequence
            Indices (TRACE_SEQUENCE_FILE) of the traces to read.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth axis.
        buffer : np.ndarray, optional
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
        """ Convert given `limits` to a `slice`. """
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
        """ Load one trace into buffer. """
        self.file_handler.xfd.gettr(buffer, index, 1, 1,
                                    limits.start, limits.stop, limits.step,
                                    buffer.size)


    # Data loading: depth slices
    def load_depth_slices(self, indices, buffer=None):
        """ Load horizontal (depth) slices of the data.
        Requires a ~full sweep through SEG-Y, therefore is slow.

        Parameters
        ----------
        indices : sequence
            Indices (ordinals) of the depth slices to read.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
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
        """ Create on iterator over the entire file traces in chunks.

        Each chunk contains no more than `chunk_size` traces.
        If `chunk_size` is not provided and `n_chunks` is given instead, there are no more than `n_chunks` chunks.
        One and only one of `chunk_size` and `n_chunks` should be provided.

        Each element in the iterator is a dictionary with `'data'`, `'start'` and `'end'` keys.

        Parameters
        ----------
        chunk_size : int, optional
            Maximum size of the chunk.
        n_chunks : int, optional
            Maximum number of chunks.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth (last) axis. Passed directly to :meth:`load_traces`.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies. Passed directly to :meth:`load_traces`.

        Returns
        -------
        iterator, info : tuple with two elements

        iterator : iterable
            An iterator over the entire SEG-Y traces.
            Each element in the iterator is a dictionary with `'data'`, `'start'` and `'end'` keys.
        info : dict
            Description of the iterator with `'chunk_size'`, `'n_chunks'`, `'chunk_starts'` and `'chunk_ends'` keys.
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
        """ A shorthand for :meth:`make_chunk_iterator` with no info returned. """
        return self.make_chunk_iterator(chunk_size=chunk_size, n_chunks=n_chunks,
                                        limits=limits, buffer=buffer)[0]


    # Inner workings
    def __enter__(self):
        return self

    def __exit__(self, _, __, ___):
        self.file_handler.close()

    def __getstate__(self):
        """ Create pickling state from `__dict__` by setting SEG-Y file handler to `None`. """
        state = copy(self.__dict__)
        state["file_handler"] = None
        return state

    def __setstate__(self, state):
        """ Recreate instance from unpickled state and reopen source SEG-Y file. """
        self.__dict__ = state
        self.file_handler = segyio.open(self.path, mode='r', endian=self.endian,
                                        strict=self.strict, ignore_geometry=self.ignore_geometry)
        self.file_handler.mmap()

    def __del__(self):
        """ Close SEG-Y file handler on loader destruction. """
        self.file_handler.close()


class SafeSegyioLoader(SegyioLoader):
    """ A thin wrapper around `segyio` library for convenient loading of headers and traces.

    Unlike :class:`SegyioLoader`, uses only public APIs to load traces.

    Used mainly for performance measurements.
    """
    def load_trace(self, index, buffer, limits):
        """ Load one trace into buffer. """
        buffer[:] = self.file_handler.trace.raw[index][limits]

    def load_depth_slice(self, index, buffer):
        """ Load one depth slice into buffer. """
        buffer[:] = self.file_handler.depth_slice[index]
