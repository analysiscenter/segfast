import numpy as np
from .file_handler import BaseMemmapHandler

class MemmapWriter(BaseMemmapHandler):
    def __init__(self, path, traces_header_spec, n_traces, n_samples, format=5, endian='big'):
        super().__init__(path, endian)

        self.traces_header_spec = self.make_headers_specs(traces_header_spec)
        self.samples_dtype = np.dtype(self.endian_symbol + self.SEGY_FORMAT_TO_TRACE_DATA_DTYPE[format])

        traces_spec = np.dtype(self._make_mmap_headers_dtype(self.traces_header_spec) + [('data', self.samples_dtype, n_samples)])
        binary_header_spec = np.dtype(self._make_mmap_binary_header_dtype())

        self.file_dtype = [
            ('textual_header', np.void, self.TEXTUAL_HEADER_LENGTH),
            ('binary_header', binary_header_spec),
            ('data', traces_spec, n_traces)
        ]

        self.mmap = np.memmap(filename=path, mode='w+', shape=1, dtype=self.file_dtype)[0]
        self.mmap['binary_header']['Format'] = format
        self.mmap['binary_header']['Traces'] = n_traces
        self.mmap['binary_header']['Samples'] = n_samples

    def set_binary_header(self, **headers):
        for name, value in headers.items():
            self.mmap['binary_header'][name] = value

    def set_traces(self, traces_data, traces_headers, indices=None):
        for i in range(len(traces_data)):
            self.mmap['data'][i]['data'] = traces_data[i]
            for name in traces_headers[i]:
                self.mmap['data'][i][name] = traces_headers[i][name]

    def close(self):
        pass
