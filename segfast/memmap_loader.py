""" Class to load headers/traces from SEG-Y via memory mapping. """

import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import dill

import numpy as np
import pandas as pd
from numba import njit, prange


from .segyio_loader import SegyioLoader
from .trace_header_spec import TraceHeaderSpec
from .utils import Notifier, ForPoolExecutor



class MemmapLoader(SegyioLoader):
    """ Custom reader/writer for SEG-Y files.
    Relies on a memory mapping mechanism for actual reads of headers and traces.

    SEG-Y description
    -----------------
    The SEG-Y is a binary file divided into several blocks:

        - file-wide information block which in most cases takes the first 3600 bytes:

            - **textual header**: the first 3200 bytes are reserved for textual info about the file. Most of the
              software uses this header to keep acquisition meta, date of creation, author, etc.
            - **binary header**: 3200–3600 bytes contain file-wide headers, which describe the number of traces,
              a format used for storing numbers, the number of samples for each trace, acquisition parameters, etc.
            - (optional) 3600+ bytes can be used to store the **extended textual information**. If there is
              such a header, then this is indicated by the value in one of the 3200–3600 bytes.

        - a sequence of traces, where each trace is a combination of its header and signal data:

            - **trace header** takes the first 240 bytes and describes the meta info about its trace: shot/receiver
              coordinates, the method of acquisition, current trace length, etc. Analogously to binary file header,
              each trace also can have extended headers.
            - **trace data** is usually an array of amplitude values, which can be stored in various numerical types.
              As the original SEG-Y is quite old (1975), one of those numerical formats is IBM float,
              which is very different from standard IEEE floats; therefore, a special caution is required to
              correctly decode values from such files.

    For the most part, SEG-Y files are written with a constant size of each trace, although the standard itself allows
    for variable-sized traces. We do not work with such files.

    Implementation details
    ----------------------
    We rely on :mod:`segyio` to infer file-wide parameters. For headers and traces, we use custom methods of reading
    binary data. Main differences to :mod:`segyio` `C++` implementation:
        - we read all of the requested headers in one file-wide sweep, speeding up by an order of magnitude
          compared to the :mod:`segyio` sequential read of every requested header.
          Also, we do that in multiple processes across chunks.

        - a memory map over trace data is used for loading values. Avoiding redundant copies and leveraging
          :mod:`numpy` superiority allows to speed up reading, especially in case of trace slicing along the samples
          axis. This is extra relevant in the case of loading horizontal (depth) slices.
    """
    def __init__(self, path, endian='big', strict=False, ignore_geometry=True):
        # Re-use most of the file-wide attributes from the `segyio` loader
        super().__init__(path=path, endian=endian, strict=strict, ignore_geometry=ignore_geometry)

        # Endian symbol for creating `numpy` dtypes
        self.endian_symbol = self.ENDIANNESS_TO_SYMBOL[endian]

        # Prefix attributes with `file`/`mmap` to avoid confusion.
        # TODO: maybe, add `segy` prefix to the attributes of the base class?
        self.file_format = self.metrics['format']
        self.file_traces_offset = self.metrics['trace0']
        self.file_trace_size = self.metrics['trace_bsize']

        # Dtype for data of each trace
        mmap_trace_data_dtype = self.SEGY_FORMAT_TO_TRACE_DATA_DTYPE[self.file_format]
        mmap_trace_data_dtype = self.endian_symbol + mmap_trace_data_dtype
        self.mmap_trace_data_dtype = mmap_trace_data_dtype
        self.mmap_trace_data_size = self.n_samples if self.file_format != 1 else (self.n_samples, 4)

        # Dtype of each trace
        # TODO: maybe, use `np.uint8` as dtype instead of `np.void` for headers as it has nicer repr
        self.mmap_trace_dtype = np.dtype([('headers', np.void, TraceHeaderSpec.TRACE_HEADER_SIZE),
                                          ('data', self.mmap_trace_data_dtype, self.mmap_trace_data_size)])
        self.data_mmap = self._construct_data_mmap()

    def _construct_data_mmap(self):
        """ Create a memory map with the first 240 bytes (headers) of each trace skipped. """
        return np.memmap(filename=self.path, mode='r', shape=self.n_traces, dtype=self.mmap_trace_dtype,
                         offset=self.file_traces_offset)["data"]


    # Headers loading
    def load_headers(self, headers, indices=None, reconstruct_tsf=True, sort_columns=True, return_specs=False,
                     chunk_size=25_000, max_workers=4, pbar=False, **kwargs):
        """ Load requested trace headers from a SEG-Y file for each trace into a dataframe.
        If needed, we reconstruct the ``'TRACE_SEQUENCE_FILE'`` manually be re-indexing traces.

        Under the hood, we create a memory mapping over the SEG-Y file, and view it with special dtype.
        That dtype skips all of the trace data bytes and all of the unrequested headers, leaving only passed `headers`
        as non-void dtype.

        The file is read in chunks in multiple processes.

        Parameters
        ----------
        headers : sequence
            An array-like where each element can be:
                - ``str`` -- header name,
                - ``int`` -- header starting byte,
                - :class:`~.trace_header_spec.TraceHeaderSpec` -- used as is,
                - ``tuple`` -- args to init :class:`~.trace_header_spec.TraceHeaderSpec`,
                - ``dict`` -- kwargs to init :class:`~.trace_header_spec.TraceHeaderSpec`.
        indices : sequence or None
            Indices of traces to load trace headers for. If not given, trace headers are loaded for all traces.
        reconstruct_tsf : bool
            Whether to reconstruct `TRACE_SEQUENCE_FILE` manually.
        sort_columns : bool
            Whether to sort columns in the resulting dataframe by their starting bytes.
        return_specs : bool
            Whether to return header specs used to load trace headers.
        chunk_size : int
            Maximum amount of traces in each chunk.
        max_workers : int, optional
            Maximum number of parallel processes to spawn. If ``None``, then the number of CPU cores is used.
        pbar : bool or str
            If bool, then whether to display progress bar over the file sweep.
            If str, then type of progress bar to display: ``'t'`` for textual, ``'n'`` for widget.

        Return
        ------
        ``pd.DataFrame``

        Examples
        --------
        * Standard ``'CDP_X'`` and ``'CDP_Y'`` headers:

        .. code-block:: python

            segfast_file.load_headers(['CDP_X', 'CDP_Y'])

        * Standard headers from 181 and 185 bytes with standard dtypes:

        .. code-block:: python

            segfast_file.load_headers([181, 185])

        * Load ``'CDP_X'`` and ``'CDP_Y'`` from non-standard bytes positions corresponding to some standard headers
          (i.e. load ``'CDP_X'`` from bytes for ``'INLINE_3D'`` with ``'<i4'`` dtype and ``'CDP_Y'`` from bytes
          for ``'CROSSLINE_3D'``):

        .. code-block:: python

            segfast_file.load_headers([{'name': 'CDP_X', 'start_byte': 189, 'dtype': '<i4'}, ('CDP_Y', 193)])

        * Load ``'CDP_X'`` and ``'CDP_Y'`` from arbitrary positions:

        .. code-block:: python

            segfast_file.load_headers([('CDP_X', 45, '>f4'), ('CDP_Y', 10, '>f4')])

        * Load 'FieldRecord' header for the first 5 traces:

        .. code-block:: python

            segfast_file.load_headers(['FieldRecord'], indices=np.arange(5))

        """
        _ = kwargs
        headers = self.make_headers_specs(headers)

        if reconstruct_tsf:
            headers = [header for header in headers if header.name != 'TRACE_SEQUENCE_FILE']

        # Construct mmap dtype: detailed for headers
        mmap_trace_headers_dtype = self._make_mmap_headers_dtype(headers)
        mmap_trace_dtype = np.dtype([*mmap_trace_headers_dtype,
                                     ('data', self.mmap_trace_data_dtype, self.mmap_trace_data_size)])

        dst_headers_dtype = [(header.name, header.dtype.str) for header in headers]
        dst_headers_dtype = np.dtype(dst_headers_dtype).newbyteorder("=")

        # Calculate the number of requested traces, chunks and a list of trace indices/slices for each chunk
        n_traces = self.n_traces if indices is None else len(indices)
        n_chunks, last_chunk_size = divmod(n_traces, chunk_size)
        if last_chunk_size:
            n_chunks += 1
        chunk_slices_list = [slice(i * chunk_size, (i + 1) * chunk_size) for i in range(n_chunks)]

        # Process `max_workers` and select executor
        max_workers = os.cpu_count() if max_workers is None else max_workers
        max_workers = min(n_chunks, max_workers)
        executor_class = ForPoolExecutor if max_workers == 1 else ProcessPoolExecutor

        # Iterate over chunks
        buffer = np.empty(shape=n_traces, dtype=dst_headers_dtype)

        with Notifier(pbar, total=n_traces) as progress_bar:
            with executor_class(max_workers=max_workers) as executor:
                def callback(future, start):
                    chunk_headers = future.result()
                    chunk_size = len(chunk_headers)
                    buffer[start : start + chunk_size] = chunk_headers
                    progress_bar.update(chunk_size)

                for i, chunk_slice in enumerate(chunk_slices_list):
                    chunk_indices = indices[chunk_slice] if indices is not None else chunk_slice
                    future = executor.submit(read_chunk, path=self.path, shape=self.n_traces,
                                             offset=self.file_traces_offset, mmap_dtype=mmap_trace_dtype,
                                             buffer_dtype=dst_headers_dtype, headers=headers, indices=chunk_indices)

                    future.add_done_callback(partial(callback, start=i * chunk_size))

        # Convert to pd.DataFrame, optionally add TSF and sort
        dataframe = pd.DataFrame(buffer, copy=False)
        dataframe, headers = self.postprocess_headers_dataframe(dataframe, headers=headers, indices=indices,
                                                                reconstruct_tsf=reconstruct_tsf,
                                                                sort_columns=sort_columns)
        if return_specs:
            return dataframe, headers
        return dataframe

    @staticmethod
    def _make_mmap_headers_dtype(headers):
        """ Create a  list of :mod:`numpy` dtypes to view headers data.

        Defines a dtype for exactly 240 bytes, where each of the requested headers would have its own named subdtype,
        and the rest of bytes are lumped into :class:`numpy.void` of certain lengths.

        Only the header data should be viewed under this dtype: the rest of trace data (values)
        should be processed (or skipped) separately.

        We do not apply the final conversion to :class:`numpy.dtype` to the resulting list of dtypes so it is easier
        to append to it.

        Examples
        --------
        If ``headers`` are ``'INLINE_3D'`` and ``'CROSSLINE_3D'``, which are 189-192 and 193-196 bytes, the output
        would be:

        .. code-block:: python

            [('unused_0', numpy.void, 188),
             ('INLINE_3D', '>i4'),
             ('CROSSLINE_3D', '>i4'),
             ('unused_1', numpy.void, 44)]
        """
        headers = sorted(headers, key=lambda x: x.start_byte)

        if headers[0].start_byte != 1:
            dtype_list = [('unused_0', np.void, headers[0].start_byte - 1)]
            unused_counter = 1
        else:
            dtype_list = []
            unused_counter = 0

        for i, header in enumerate(headers):
            header_dtype = (header.name, str(header.dtype))
            dtype_list.append(header_dtype)

            next_byte_position = headers[i+1].start_byte \
                                 if i + 1 < len(headers) \
                                 else TraceHeaderSpec.TRACE_HEADER_SIZE + 1

            unused_len = next_byte_position - (header.start_byte + header.byte_len)
            if unused_len > 0:
                unused_header = (f'unused_{unused_counter}', np.void, unused_len)
                dtype_list.append(unused_header)
                unused_counter += 1
            elif unused_len < 0:
                raise ValueError(f'{header.name} header overlap')

        return dtype_list


    # Traces loading
    def load_traces(self, indices, limits=None, buffer=None):
        """ Load traces by their indices.
        Under the hood, we use pre-made memory mapping over the file, where trace data is viewed with a special dtype.
        Regardless of the numerical dtype of SEG-Y file, we output IEEE float32:
        for IBM floats, that requires an additional conversion.

        Parameters
        ----------
        indices : sequence
            Indices (``'TRACE_SEQUENCE_FILE'``) of the traces to read.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth axis.
        buffer : numpy.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.

        Return
        ------
        numpy.ndarray
        """
        limits = self.process_limits(limits)

        if self.file_format != 1:
            traces = self.data_mmap[indices, limits]
        else:
            traces = self.data_mmap[indices, limits.start:limits.stop]
            if limits.step != 1:
                traces = traces[:, ::limits.step]
            traces = self._ibm_to_ieee(traces)

        if buffer is None:
            return np.require(traces, dtype=self.dtype, requirements='C')

        buffer[:len(indices)] = traces
        return buffer

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
        depth_slices = self.data_mmap[:, indices]
        if self.file_format == 1:
            depth_slices = self._ibm_to_ieee(depth_slices)
        depth_slices = depth_slices.T

        if buffer is None:
            return np.require(depth_slices, dtype=np.float32, requirements='C')
        buffer[:] = depth_slices
        return buffer

    def _ibm_to_ieee(self, array):
        """ Convert IBM floats to regular IEEE ones. """
        array_bytes = (array[:, :, 0], array[:, :, 1], array[:, :, 2], array[:, :, 3])
        if self.endian in {"little", "lsb"}:
            array_bytes = array_bytes[::-1]
        return ibm_to_ieee(*array_bytes)


    # Inner workings
    def __getstate__(self):
        """ Create pickling state from `__dict__` by setting SEG-Y file handler and memmap to `None`. """
        state = super().__getstate__()
        state["data_mmap"] = None
        return state

    def __setstate__(self, state):
        """ Recreate instance from the unpickled state, reopen source SEG-Y file and memmap. """
        super().__setstate__(state)
        self.data_mmap = self._construct_data_mmap()


    # Conversion to other SEG-Y formats (data dtype)
    def convert(self, path=None, format=8, transform=None, chunk_size=25_000, max_workers=4,
                pbar='t', overwrite=True):
        """ Convert SEG-Y file to a different ``format``: dtype of data values.
        Keeps the same binary header (except for the 3225 byte, which stores the format).
        Keeps the same header values for each trace: essentially, only the values of each trace are transformed.

        The most common scenario of this function usage is to convert float32 SEG-Y into int8 one:
        the latter is a lot faster and takes ~4 times less disk space at the cost of some data loss.

        Parameters
        ----------
        path : str, optional
            Path to the save file to. If not provided, we use the path of the current cube with an added postfix.
        format : int
            Target SEG-Y format.
            Refer to :attr:`.SEGY_FORMAT_TO_TRACE_DATA_DTYPE` for list of available formats and their data value dtype.
        transform : callable, optional
            Callable to transform data from the current file to the ones, saved in ``path``.
            Must return the same dtype, as specified by ``format``.
        chunk_size : int
            Maximum amount of traces in each chunk.
        max_workers : int or None
            Maximum number of parallel processes to spawn. If None, then the number of CPU cores is used.
        pbar : bool, str
            If bool, then whether to display a progress bar.
            If str, then the type of progress bar to display: ``'t'`` for textual, ``'n'`` for widget.
        overwrite : bool
            Whether to overwrite the existing ``path`` or raise an exception.

        Return
        ------
        path : str
        """
        #pylint: disable=redefined-builtin
        # Default path
        if path is None:
            dirname = os.path.dirname(self.path)
            basename = os.path.basename(self.path)
            path = os.path.join(dirname, basename.replace('.', f'_f{format}.'))

        # Compute target dtype, itemsize, size of the dst file
        dst_dtype = self.endian_symbol + self.SEGY_FORMAT_TO_TRACE_DATA_DTYPE[format]
        dst_itemsize = np.dtype(dst_dtype).itemsize
        dst_size = self.file_traces_offset + \
                   self.n_traces * (TraceHeaderSpec.TRACE_HEADER_SIZE + self.n_samples * dst_itemsize)

        # Exceptions
        traces = self.load_traces([0])
        if transform(traces).dtype != dst_dtype:
            raise ValueError('dtype of `dst` is not the same as the one returned by `transform`!.'
                             f' {dst_dtype}!={transform(traces).dtype}')

        if os.path.exists(path) and not overwrite:
            raise OSError(f'File {path} already exists! Set `overwrite=True` to ignore this error.')

        # Serialize `transform`
        transform = dill.dumps(transform)

        # Create new file and copy binary header
        src_mmap = np.memmap(self.path, mode='r')
        dst_mmap = np.memmap(path, mode='w+', shape=(dst_size,))
        dst_mmap[:self.file_traces_offset] = src_mmap[:self.file_traces_offset]

        # Replace `format` bytes
        dst_mmap[3225-1:3225-1+2] = np.array([format], dtype=self.endian_symbol + 'u2').view('u1')

        # Prepare dst dtype
        dst_trace_dtype = np.dtype([('headers', np.void, TraceHeaderSpec.TRACE_HEADER_SIZE),
                                    ('data', dst_dtype, self.n_samples)])

        # Split the whole file into chunks no larger than `chunk_size`
        n_chunks, last_chunk_size = divmod(self.n_traces, chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            chunk_sizes += [last_chunk_size]
        chunk_starts = np.cumsum([0] + chunk_sizes[:-1])

        # Process `max_workers` and select executor
        max_workers = os.cpu_count() if max_workers is None else max_workers
        max_workers = min(len(chunk_sizes), max_workers)
        executor_class = ForPoolExecutor if max_workers == 1 else ProcessPoolExecutor

        # Iterate over chunks
        name = os.path.basename(path)
        with Notifier(pbar, total=self.n_traces, desc=f'Convert to `{name}`', ncols=110) as progress_bar:
            with executor_class(max_workers=max_workers) as executor:
                def callback(future):
                    chunk_size = future.result()
                    progress_bar.update(chunk_size)

                for start, chunk_size_ in zip(chunk_starts, chunk_sizes):
                    future = executor.submit(convert_chunk,
                                             src_path=self.path, dst_path=path,
                                             shape=self.n_traces, offset=self.file_traces_offset,
                                             src_dtype=self.mmap_trace_dtype, dst_dtype=dst_trace_dtype,
                                             endian=self.endian, transform=transform,
                                             start=start, chunk_size=chunk_size_)
                    future.add_done_callback(callback)
        return path


def read_chunk(path, shape, offset, mmap_dtype, buffer_dtype, headers, indices):
    """ Read headers from one chunk.
    We create memory mapping anew in each worker, as it is easier and creates no significant overhead.
    """
    mmap = np.memmap(filename=path, mode='r', shape=shape, offset=offset, dtype=mmap_dtype)
    headers_chunk = mmap[[header.name for header in headers]][indices]

    # Explicitly cast trace headers from mmap dtype to the target architecture dtype
    buffer = np.empty_like(headers_chunk, dtype=buffer_dtype)
    buffer[:] = headers_chunk
    return buffer


def convert_chunk(src_path, dst_path, shape, offset, src_dtype, dst_dtype, endian, transform, start, chunk_size):
    """ Copy the headers, transform, and write data from one chunk.
    We create all memory mappings anew in each worker, as it is easier and creates no significant overhead.
    """
    # Deserialize `transform`
    transform = dill.loads(transform)

    # Create mmaps: src is read-only, dst is read-write
    src_mmap = np.memmap(src_path, mode='r', shape=shape, offset=offset, dtype=src_dtype)
    dst_mmap = np.memmap(dst_path, mode='r+', shape=shape, offset=offset, dtype=dst_dtype)

    # Load all data from chunk
    src_traces = src_mmap[start : start + chunk_size]
    dst_traces = dst_mmap[start : start + chunk_size]

    # If `src_traces_data` is in IBM float, convert to float32
    src_traces_data = src_traces['data']
    if len(src_traces_data.shape) == 3:
        array_bytes = (src_traces_data[:, :, 0],src_traces_data[:, :, 1],
                       src_traces_data[:, :, 2], src_traces_data[:, :, 3])
        if endian in {"little", "lsb"}:
            array_bytes = array_bytes[::-1]
        src_traces_data = ibm_to_ieee(*array_bytes)

    # Copy headers, write transformed data
    dst_traces['headers'] = src_traces['headers']
    dst_traces['data'] = transform(src_traces_data)
    return chunk_size


@njit(nogil=True, parallel=True)
def ibm_to_ieee(hh, hl, lh, ll):
    """ Convert 4 arrays representing individual bytes of IBM 4-byte floats into a single array of floats.
    Input arrays are ordered from most to least significant bytes and have ``numpy.uint8`` dtypes.
    The result is returned as an ``numpy.float32`` array.
    """
    # pylint: disable=not-an-iterable
    res = np.empty_like(hh, dtype=np.float32)
    for i in prange(res.shape[0]):
        for j in prange(res.shape[1]):
            mant = (((np.int32(hl[i, j]) << 8) | lh[i, j]) << 8) | ll[i, j]
            if hh[i, j] & 0x80:
                mant = -mant
            exp16 = (np.int8(hh[i, j]) & np.int8(0x7f)) - 70
            res[i, j] = mant * 16.0**exp16
    return res
