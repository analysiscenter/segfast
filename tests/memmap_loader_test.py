import sys; sys.path.insert(0, '.')

import os
from functools import lru_cache
from contextlib import ExitStack as does_not_raise
import pytest
import pickle
import segfast

import numpy as np

POSITIONS = np.array([181, 185, 189, 193])
HEADERS_NAMES = ['CDP_X', 'CDP_Y', 'INLINE_3D', 'CROSSLINE_3D']

STANDARD_HEADERS = tuple([segfast.TraceHeaderSpec(start_byte=start_byte) for start_byte in POSITIONS])
NONSTANDARD_NAMES_HEADERS = tuple(
    [segfast.TraceHeaderSpec(name, start_byte=start_byte) for name, start_byte in zip(HEADERS_NAMES, POSITIONS - 4)]
)

NONSTANDARD_POSITIONS_HEADERS = tuple(
    [segfast.TraceHeaderSpec(name, start_byte=start_byte, dtype=dtype) for name, start_byte, dtype in zip(HEADERS_NAMES, POSITIONS - 5, ['>i4', '<i2', '>f4', '<f4'])]
)

@pytest.fixture(scope="session")
def session_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('data')

@lru_cache
def make_poststack_cube(tmp_folder, n_samples=1000, sample_interval=200, format=5, endian='big',
                        ilines=(0, 200, 2), xlines=(100, 300, 1), headers=STANDARD_HEADERS):
    path = str(tmp_folder) + '/' + str(hash((n_samples, sample_interval, format, endian, ilines, xlines, headers))) + '.sgy'
    ilines = np.arange(*ilines)
    xlines = np.arange(*xlines)
    trace_headers_spec = headers
    n_traces = len(ilines) * len(xlines)
    ilines_grid, xlines_grid = np.meshgrid(ilines, xlines, indexing='ij')

    cdp_x, cdp_y = ilines * 25, xlines * 25
    cdp_x_grid, cdp_y_grid = ilines_grid * 25, xlines_grid * 25

    data = np.ones(shape=(len(ilines) * len(xlines), n_samples))#np.random.normal(size=(len(ilines) * len(xlines), n_samples))

    trace_headers = [{
        trace_headers_spec[0].name: cdp_x_grid.flatten()[i],
        trace_headers_spec[1].name: cdp_y_grid.flatten()[i],
        trace_headers_spec[2].name: ilines_grid.flatten()[i],
        trace_headers_spec[3].name: xlines_grid.flatten()[i],
    } for i in range(n_traces)]

    sgy = segfast.MemmapWriter(path=path, n_traces=n_traces, n_samples=n_samples, format=format,
                               endian=endian,
                               traces_header_spec=trace_headers_spec)
    sgy.set_binary_header(Interval=sample_interval)
    sgy.set_traces(data, trace_headers)

    return {
        'path': path,
        'inlines': ilines,
        'crosslines': xlines,
        'n_samples': n_samples,
        'delay': 0,
        'headers': trace_headers_spec,
        'endian': endian,
        'cdp_x': cdp_x,
        'cdp_y': cdp_y,
        'sample_interval': sample_interval
    }



@pytest.mark.parametrize('engine', ['memmap', 'segyio', segfast.MemmapLoader, segfast.SegyioLoader, 'foo'])
@pytest.mark.parametrize('endian', ['little', 'big'])
def test_engine(session_dir, engine, endian):
    params = make_poststack_cube(session_dir, endian=endian)

    expectation = does_not_raise() if engine != 'foo' else pytest.raises(ValueError)
    with expectation:
        file = segfast.open(params['path'], endian=params['endian'], engine=engine)

        assert file.sample_interval == params['sample_interval']
        assert file.delay == params['delay']

@pytest.mark.parametrize('header', ['CDP_X', 181, ('CDP_X', 181), ('CDP_X', 181, 'i4'), {'name': 'CDP_X'}])
def test_make_headers(session_dir, header):
    params = make_poststack_cube(session_dir)
    path = params['path']

    with segfast.open(path, endian=params['endian']) as file:
        header_df = file.load_header(header)
        assert (np.unique(header_df['CDP_X']) == params['cdp_x']).all()

@pytest.mark.parametrize('engine', ['memmap', 'segyio'])
@pytest.mark.parametrize('endian', ['little', 'big'])
class TestLoader:
    @pytest.mark.parametrize('pbar', [False, True])
    @pytest.mark.parametrize('sort_columns', [False, True])
    @pytest.mark.parametrize('headers', [STANDARD_HEADERS, NONSTANDARD_NAMES_HEADERS, NONSTANDARD_POSITIONS_HEADERS])
    def test_headers(self, session_dir, engine, endian, headers, pbar, sort_columns):
        params = make_poststack_cube(session_dir, headers=headers, endian=endian)
        path = params['path']

        with segfast.open(path, endian=params['endian'], engine=engine) as file:
            expectation = pytest.raises(KeyError) if engine == 'segyio' and headers == NONSTANDARD_POSITIONS_HEADERS else does_not_raise()
            with expectation:
                headers_df = file.load_headers(params['headers'], sort_columns=sort_columns, pbar=pbar)

                assert len(headers_df) == len(params['inlines']) * len(params['crosslines'])

                for column_name, param_name in zip([item.name for item in headers], ['cdp_x', 'cdp_y', 'inlines', 'crosslines']):
                    assert (np.unique(headers_df[column_name]) == params[param_name]).all()

                assert set(headers_df.columns) == set([*[item.name for item in file.make_headers_specs(params['headers'])], 'TRACE_SEQUENCE_FILE'])

    def test_all_headers(self, session_dir, engine, endian):
        params = make_poststack_cube(session_dir, endian=endian)
        path = params['path']

        with segfast.open(path, endian=params['endian'], engine=engine) as file:
            _ = file.load_headers('all')

    def test_overlaped_headers(self, session_dir, engine, endian):
        params = make_poststack_cube(session_dir, endian=endian)
        path = params['path']

        with segfast.open(path, endian=params['endian'], engine=engine) as file:
            expectation = pytest.raises(ValueError) if engine != 'segyio' else pytest.raises(KeyError)
            with expectation:
                _ = file.load_headers([('A', 20, 'i4'), ('B', 22, 'i4')])

    @pytest.mark.parametrize('buffer', [False, True])
    def test_traces(self, session_dir, buffer, engine, endian):
        params = make_poststack_cube(session_dir, endian=endian)
        path = params['path']

        with segfast.open(path, endian=params['endian'], engine=engine) as file:
            buffer = np.empty((file.n_traces, file.n_samples), dtype='float32') if buffer else None
            values = file.load_traces(indices=list(range(file.n_traces)), buffer=buffer)

            assert (values == 1).all()

    @pytest.mark.parametrize('buffer', [False, True])
    def test_depth_slices(self, session_dir, buffer, engine, endian):
        params = make_poststack_cube(session_dir, endian=endian)
        path = params['path']

        with segfast.open(path, endian=params['endian'], engine=engine) as file:
            indices = [file.n_samples // 2, file.n_samples // 2 + 10]
            buffer = np.empty((len(indices), file.n_traces), dtype='float32') if buffer else None
            values = file.load_depth_slices(indices=indices, buffer=buffer)

            assert (values == 1).all()


    def test_pickling(self, session_dir, engine, endian):
        params = make_poststack_cube(session_dir, endian=endian)
        path = params['path']

        file = segfast.open(path, endian=params['endian'], engine=engine)
        obj = pickle.dumps(file)
        recreated_file = pickle.loads(obj)

        assert (file.load_header('CDP_X') == recreated_file.load_header('CDP_X')).all().all()

    @pytest.mark.parametrize('max_workers', [1, 4])
    @pytest.mark.parametrize('format', [5, 8])
    @pytest.mark.parametrize('path', [False, True])
    def test_convert(self, session_dir, max_workers, engine, endian, format, path):
        params = make_poststack_cube(session_dir, endian=endian)
        path = params['path']

        with segfast.open(path, endian=params['endian'], engine=engine) as file:
            expectation = does_not_raise() if engine != 'segyio' else pytest.raises(AttributeError)
            with expectation:
                transform = (lambda x: x) if format == 5 else (lambda x: x.astype('int8'))
                data = file.load_traces(np.arange(file.n_traces))
                new_path = os.path.splitext(path)[0] + f'_converted_{format}.py' + '' if path else None
                new_path = file.convert(path=new_path, format=format, transform=transform, max_workers=max_workers, pbar='t', overwrite=True)

                new_file = segfast.open(new_path, endian=params['endian'], engine=engine)
                assert (new_file.load_headers(params['headers']) == file.load_headers(params['headers'])).all().all()

                new_data = new_file.load_traces(np.arange(new_file.n_traces))
                assert (transform(data) == new_data).all()
