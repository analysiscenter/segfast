""" Class to load headers/traces from SEG-Y via memory mapping. """
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import dill

import numpy as np
import pandas as pd
from numba import njit, prange

import segyio


from .segyio_loader import SegyioLoader
from .utils import Notifier, ForPoolExecutor



class MemmapLoader(SegyioLoader):
    """ Custom reader/writer for SEG-Y files.
    Relies on memory mapping mechanism for actual reads of headers and traces.

    SEG-Y description
    -----------------
    Here we give a brief intro into SEG-Y format. Each SEG-Y file consists of:
        - file-wide information, in most cases the first 3600 bytes.
            - the first 3200 bytes are reserved for textual info about the file.
            Most software uses this to keep track of processing operations, date of creation, author, etc.
            - 3200-3600 bytes contain file-wide headers, which describe the number of traces,
            used format, depth, acquisition parameters, etc.
            - 3600+ bytes can be used to store the extended textual information, which is optional and indicated by
            one of the values in 3200-3600 bytes.

        - a sequence of traces, where each trace is a combination of header and its actual data.
            - header is the first 240 bytes and it describes the meta info about that trace:
            its coordinates in different types, the method of acquisition, etc.
            - data is an array of values, usually amplitudes, which can be stored in multiple numerical types.
            As the original SEG-Y is quite old (1975), one of those numerical formats is IBM float,
            which is very different from standard IEEE floats; therefore, a special caution is required to
            correctly decode values from such files.

    For the most part, SEG-Y files are written with constant size of each trace, although the standard itself allows
    for variable-sized traces. We do not work with such files.


    Implementation details
    ----------------------
    We rely on `segyio` to infer file-wide parameters.

    For headers and traces, we use custom methods of reading binary data.
    Main differences to `segyio C++` implementation:
        - we read all of the requested headers in one file-wide sweep, speeding up by an order of magnitude
        compared to the `segyio` sequential read of every requested header.
        Also, we do that in multiple processes across chunks.

        - a memory map over traces data is used for loading values. Avoiding redundant copies and leveraging
        `numpy` superiority allows to speed up reading, especially in case of trace slicing along the samples axis.
        This is extra relevant in case of loading horizontal (depth) slices.
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
        self.mmap_trace_dtype = np.dtype([('headers', np.void, self.TRACE_HEADER_SIZE),
                                          ('data', self.mmap_trace_data_dtype, self.mmap_trace_data_size)])
        self.data_mmap = self._construct_data_mmap()

    def _construct_data_mmap(self):
        """ Create a memory map with the first 240 bytes (headers) of each trace skipped. """
        return np.memmap(filename=self.path, mode='r', shape=self.n_traces, dtype=self.mmap_trace_dtype,
                         offset=self.file_traces_offset)["data"]


    # Headers
    def load_headers(self, headers, chunk_size=25_000, max_workers=4, pbar=False,
                     reconstruct_tsf=True, sort_columns=True, **kwargs):
        """ Load requested trace headers from a SEG-Y file for each trace into a dataframe.
        If needed, we reconstruct the `'TRACE_SEQUENCE_FILE'` manually be re-indexing traces.

        Under the hood, we create a memory mapping over the SEG-Y file, and view it with a special dtype.
        That dtype skips all of the trace data bytes and all of the unrequested headers, leaving only passed `headers`
        as non-void dtype.

        The file is read in chunks in multiple processes.

        Parameters
        ----------
        headers : sequence
            Names of headers to load.
        chunk_size : int
            Maximum amount of traces in each chunk.
        max_workers : int or None
            Maximum number of parallel processes to spawn. If None, then the number of CPU cores is used.
        pbar : bool, str
            If bool, then whether to display progress bar over the file sweep.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        reconstruct_tsf : bool
            Whether to reconstruct `TRACE_SEQUENCE_FILE` manually.
        """
        _ = kwargs
        headers = list(headers)

        if reconstruct_tsf and 'TRACE_SEQUENCE_FILE' in headers:
            headers.remove('TRACE_SEQUENCE_FILE')

        # Construct mmap dtype: detailed for headers
        mmap_trace_headers_dtype = self._make_mmap_headers_dtype(headers, endian_symbol=self.endian_symbol)
        mmap_trace_dtype = np.dtype([*mmap_trace_headers_dtype,
                                     ('data', self.mmap_trace_data_dtype, self.mmap_trace_data_size)])

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
        buffer = np.empty((self.n_traces, len(headers)), dtype=np.int32)

        with Notifier(pbar, total=self.n_traces) as progress_bar:
            with executor_class(max_workers=max_workers) as executor:

                def callback(future, start):
                    chunk_headers = future.result()
                    chunk_size = len(chunk_headers)
                    buffer[start : start + chunk_size] = chunk_headers
                    progress_bar.update(chunk_size)

                for start, chunk_size_ in zip(chunk_starts, chunk_sizes):
                    future = executor.submit(read_chunk, path=self.path,
                                             shape=self.n_traces, offset=self.file_traces_offset,
                                             dtype=mmap_trace_dtype, headers=headers,
                                             start=start, chunk_size=chunk_size_)
                    future.add_done_callback(partial(callback, start=start))

        # Convert to pd.DataFrame, optionally add TSF and sort
        dataframe = pd.DataFrame(buffer, columns=headers, copy=False)
        dataframe = self.postprocess_headers_dataframe(dataframe, headers=headers,
                                                       reconstruct_tsf=reconstruct_tsf, sort_columns=sort_columns)
        return dataframe

    def load_header(self, header, chunk_size=25_000, max_workers=None, pbar=False, **kwargs):
        """ Load exactly one header. """
        return self.load_headers(headers=[header], chunk_size=chunk_size, max_workers=max_workers,
                                 pbar=pbar, reconstruct_tsf=False, **kwargs)

    @staticmethod
    def _make_mmap_headers_dtype(headers, endian_symbol='>'):
        """ Create list of `numpy` dtypes to view headers data.

        Defines a dtype for exactly 240 bytes, where each of the requested headers would have its own named subdtype,
        and the rest of bytes are lumped into `np.void` of certain lengths.

        Only the headers data should be viewed under this dtype: the rest of trace data (values)
        should be processed (or skipped) separately.

        We do not apply final conversion to `np.dtype` to the resulting list of dtypes so it is easier to append to it.

        Examples
        --------
        if `headers` are `INLINE_3D` and `CROSSLINE_3D`, which are 189-192 and 193-196 bytes, the output would be:
        >>> [('unused_0', numpy.void, 188),
        >>>  ('INLINE_3D', '>i4'),
        >>>  ('CROSSLINE_3D', '>i4'),
        >>>  ('unused_1', numpy.void, 44)]
        """
        header_to_byte = segyio.tracefield.keys
        byte_to_header = {val: key for key, val in header_to_byte.items()}
        start_bytes = sorted(header_to_byte.values())
        byte_to_len = {start: end - start
                       for start, end in zip(start_bytes, start_bytes[1:] + [MemmapLoader.TRACE_HEADER_SIZE + 1])}
        requested_headers_bytes = {header_to_byte[header] for header in headers}

        # Iterate over all headers
        # Unrequested headers are lumped into `np.void` of certain lengths
        # Requested   headers are each its own dtype
        dtype_list = []
        unused_counter, void_counter = 0, 0
        for byte, header_len in byte_to_len.items():
            if byte in requested_headers_bytes:
                if void_counter:
                    unused_dtype = (f'unused_{unused_counter}', np.void, void_counter)
                    dtype_list.append(unused_dtype)

                    unused_counter += 1
                    void_counter = 0

                header_name = byte_to_header[byte]
                value_dtype = 'i2' if header_len == 2 else 'i4'
                value_dtype = endian_symbol + value_dtype
                header_dtype = (header_name, value_dtype)
                dtype_list.append(header_dtype)
            else:
                void_counter += header_len

        if void_counter:
            unused_dtype = (f'unused_{unused_counter}', np.void, void_counter)
            dtype_list.append(unused_dtype)
        return dtype_list


    # Data loading
    def load_traces(self, indices, limits=None, buffer=None):
        """ Load traces by their indices.
        Under the hood, we use a pre-made memory mapping over the file, where trace data is viewed with a special dtype.
        Regardless of the numerical dtype of SEG-Y file, we output IEEE float32:
        for IBM floats, that requires an additional conversion.

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
        Requires a ~full sweep through SEG-Y, therefore is slow.

        Parameters
        ----------
        indices : sequence
            Indices (ordinals) of the depth slices to read.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
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


    # Inner workingss
    def __getstate__(self):
        """ Create pickling state from `__dict__` by setting SEG-Y file handler and memmap to `None`. """
        state = super().__getstate__()
        state["data_mmap"] = None
        return state

    def __setstate__(self, state):
        """ Recreate instance from unpickled state, reopen source SEG-Y file and memmap. """
        super().__setstate__(state)
        self.data_mmap = self._construct_data_mmap()


    # Conversion to other SEG-Y formats (data dtype)
    def convert(self, path=None, format=8, transform=None, chunk_size=25_000, max_workers=4,
                pbar='t', overwrite=True):
        """ Convert SEG-Y file to a different `format`: dtype of data values.
        Keeps the same binary header (except for the 3225 byte, which stores the format).
        Keeps the same header values for each trace: essentially, only the values of each trace are transformed.

        The most common scenario of this function usage is to convert float32 SEG-Y into int8 one:
        the latter is a lot faster and takes ~4 times less disk space at the cost of some data loss.

        Parameters
        ----------
        path : str, optional
            Path to save file to. If not provided, we use the path of the current cube with an added postfix.
        format : int
            Target SEG-Y format.
            Refer to :attr:`SEGY_FORMAT_TO_TRACE_DATA_DTYPE` for list of available formats and their data value dtype.
        transform : callable, optional
            Callable to transform data from the current file to the ones, saved in `path`.
            Must return the same dtype, as specified by `format`.
        chunk_size : int
            Maximum amount of traces in each chunk.
        max_workers : int or None
            Maximum number of parallel processes to spawn. If None, then the number of CPU cores is used.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        overwrite : bool
            Whether to overwrite existing `path` or raise an exception.
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
        dst_size = self.file_traces_offset + self.n_traces * (self.TRACE_HEADER_SIZE + self.n_samples * dst_itemsize)

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
        dst_trace_dtype = np.dtype([('headers', np.void, self.TRACE_HEADER_SIZE),
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


def read_chunk(path, shape, offset, dtype, headers, start, chunk_size):
    """ Read headers from one chunk.
    We create memory mapping anew in each worker, as it is easier and creates no significant overhead.
    """
    # mmap is created over the entire file as
    # creating data over the requested chunk only does not speed up anything
    mmap = np.memmap(filename=path, mode='r', shape=shape, offset=offset, dtype=dtype)

    buffer = np.empty((chunk_size, len(headers)), dtype=np.int32)
    for i, header in enumerate(headers):
        buffer[:, i] = mmap[header][start : start + chunk_size]
    return buffer


def convert_chunk(src_path, dst_path, shape, offset, src_dtype, dst_dtype, endian, transform, start, chunk_size):
    """ Copy the headers, transform and write data from one chunk.
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
    Input arrays are ordered from most to least significant bytes and have `np.uint8` dtypes.
    The result is returned as an `np.float32` array.
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
