import numpy as np
import segyio

from .utils import TraceHeaderSpec

class BaseFileHandler:
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

    TEXTUAL_HEADER_LENGTH = 3200
    BINARY_HEADER_LENGTH = 400

    def __init__(self, path, endian='>'):
        # Parse arguments for errors
        if endian not in self.ENDIANNESS_TO_SYMBOL:
            raise ValueError(f'Unknown endian {endian}, must be one of {self.ENDIANNESS_TO_SYMBOL}')

        # Store arguments
        self.path = path
        self.endian = endian

        # Endian symbol for creating `numpy` dtypes
        self.endian_symbol = self.ENDIANNESS_TO_SYMBOL[endian]

    def make_headers_specs(self, headers):
        """ Make instances of TraceHeaderSpec. """
        byteorder = self.ENDIANNESS_TO_SYMBOL[self.endian]

        if headers == 'all':
            return [TraceHeaderSpec(start_byte, byteorder=byteorder)
                    for start_byte in TraceHeaderSpec.STANDARD_BYTE_TO_HEADER]

        headers_ = []
        for header in headers:
            if isinstance(header, TraceHeaderSpec):
                headers_.append(header.set_default_byteorder(byteorder))
            else:
                if isinstance(header, int):
                    header = (None, header)
                if not isinstance(header, (list, tuple, dict)):
                    header = (header,)
                if isinstance(header, dict):
                    init_kwargs = header
                else:
                    init_kwargs = dict(zip(['name', 'start_byte', 'dtype'], header))
                init_kwargs = {'byteorder': byteorder, **init_kwargs}
                headers_.append(TraceHeaderSpec(**init_kwargs))
        return headers_

    def make_tsf_header(self):
        """ Reconstruct the `TRACE_SEQUENCE_FILE` header. """
        dtype = np.int32 if self.n_traces < np.iinfo(np.int32).max else np.int64
        return np.arange(1, self.n_traces + 1, dtype=dtype)

    def postprocess_headers_dataframe(self, dataframe, headers, reconstruct_tsf=True, sort_columns=True):
        """ Optionally add TSF header and sort columns of a headers dataframe. """
        if reconstruct_tsf:
            dataframe['TRACE_SEQUENCE_FILE'] = self.make_tsf_header()
            headers.append(TraceHeaderSpec('TRACE_SEQUENCE_FILE'))

        if sort_columns:
            headers_bytes = [item.start_byte for item in headers]
            columns = np.array([item.name for item in headers])[np.argsort(headers_bytes)]
            dataframe = dataframe[columns]
        return dataframe

class BaseMemmapHandler(BaseFileHandler):
    BINARY_HEADER_NAME_TO_BYTE = {
        'JobID': 3201,
        'LineNumber': 3205,
        'ReelNumber': 3209,
        'Traces': 3213,
        'AuxTraces': 3215,
        'Interval': 3217,
        'IntervalOriginal': 3219,
        'Samples': 3221,
        'SamplesOriginal': 3223,
        'Format': 3225,
        'EnsembleFold': 3227,
        'SortingCode': 3229,
        'VerticalSum': 3231,
        'SweepFrequencyStart': 3233,
        'SweepFrequencyEnd': 3235,
        'SweepLength': 3237,
        'Sweep': 3239,
        'SweepChannel': 3241,
        'SweepTaperStart': 3243,
        'SweepTaperEnd': 3245,
        'Taper': 3247,
        'CorrelatedTraces': 3249,
        'BinaryGainRecovery': 3251,
        'AmplitudeRecovery': 3253,
        'MeasurementSystem': 3255,
        'ImpulseSignalPolarity': 3257,
        'VibratoryPolarity': 3259,
        'ExtTraces': 3261,
        'ExtAuxTraces': 3265,
        'ExtSamples': 3269,
        'ExtInterval': 3273,
        'ExtIntervalOriginal': 3281,
        'ExtSamplesOriginal': 3289,
        'ExtEnsembleFold': 3293,
        'IntergerConstant': 3297,
        'Unassigned1': 3301,
        'SEGYRevision': 3501,
        'SEGYRevisionMinor': 3502,
        'TraceFlag': 3503,
        'ExtendedTextualHeaders': 3505,
        'MaximumAdditionalTraceHeaders': 3507,
        'TimeCode': 3511,
        'NumFileTraces': 3513,
        'TracesOffset': 3521,
        'StanzaRecords': 3529,
        'Unassigned2': 3533
    }

    def _make_mmap_binary_header_dtype(self):
        binary_header_dtype = []
        positions = np.diff(list(self.BINARY_HEADER_NAME_TO_BYTE.values()) + [3601])
        for name, byte_len in zip(self.BINARY_HEADER_NAME_TO_BYTE.keys(), positions):
            if byte_len not in [1, 2, 4, 8]:
                dtype = (np.void, byte_len)
            elif name in ('ExtInterval', 'ExtIntervalOriginal'):
                dtype = (self.endian_symbol + 'f8', )
            else:
                dtype = (self.endian_symbol + 'i' + str(byte_len), )
            binary_header_dtype.append((name, *dtype))
        return binary_header_dtype

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
        headers = sorted(headers, key=lambda x: x.start_byte)

        unused_counter = 0
        dtype_list = []
        if headers[0].start_byte != 1:
            dtype_list = [(f'unused_{unused_counter}', np.void, headers[0].start_byte - 1)]
            unused_counter += 1

        for i, header in enumerate(headers):
            header_dtype = (header.name, str(header.dtype))
            dtype_list.append(header_dtype)

            next_byte_position = headers[i+1].start_byte \
                                 if i + 1 < len(headers) \
                                 else TraceHeaderSpec.TRACE_HEADER_SIZE + 1

            unused_len = next_byte_position - header.start_byte - header.byte_len
            if unused_len > 0:
                unused_header = (f'unused_{unused_counter}', np.void, unused_len)
                dtype_list.append(unused_header)
                unused_counter += 1
            elif unused_len < 0:
                raise ValueError(f'{header.name} header overlap')

        return dtype_list
