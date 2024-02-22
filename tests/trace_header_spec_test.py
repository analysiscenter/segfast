""" Tests for TraceHeaderSpec class. """
import sys; sys.path.insert(0, '.')

from contextlib import ExitStack as does_not_raise
import pytest
from segfast import TraceHeaderSpec

@pytest.mark.parametrize('name, start_byte, dtype', [('CDP_X', 181, 'i4'),
                                                     ('CDP_Y', 185, 'i4')])
@pytest.mark.parametrize('init_by', ['name', 'start_byte'])
class TestStandardHeaders:
    def _make_spec(self, name, start_byte, init_by):
        if init_by == 'name':
            header_spec = TraceHeaderSpec(name)
        else:
            header_spec = TraceHeaderSpec(start_byte=start_byte)
        return header_spec

    def test_standard_headers(self, name, start_byte, dtype, init_by):
        header_spec = self._make_spec(name, start_byte, init_by)

        assert header_spec.name == name
        assert header_spec.start_byte == start_byte
        assert header_spec.dtype == dtype

    def test_standard_checks(self, name, start_byte, dtype, init_by):
        header_spec = self._make_spec(name, start_byte, init_by)

        assert header_spec.has_standard_location
        assert header_spec.is_standard

@pytest.mark.parametrize('name, dtype', [('CDP_X', 'i4')])
@pytest.mark.parametrize('start_byte', [181, 182])
def test_standard_name(name, start_byte, dtype):
    non_standard_name = 'NEW_HEADER'
    expectation = does_not_raise() if start_byte in TraceHeaderSpec.STANDARD_BYTE_TO_HEADER else pytest.raises(ValueError)
    with expectation:
        header_spec = TraceHeaderSpec(non_standard_name, start_byte=start_byte, dtype=dtype)

        assert header_spec.name == non_standard_name
        assert header_spec.standard_name == name

@pytest.mark.parametrize('start_byte, dtype', [(240+100, 'i4'), (240-2, 'f8')])
def test_out_of_bounds_position(start_byte, dtype):
    with pytest.raises(ValueError):
        _ = TraceHeaderSpec(name='NEW_HEADER', start_byte=start_byte, dtype=dtype)

@pytest.mark.parametrize('name, start_byte', [('NEW_HEADER', 17)])
@pytest.mark.parametrize('dtype', ['f4', '>f4'])
class TestConversions:
    def test_repr(self, name, start_byte, dtype):
        header_spec = TraceHeaderSpec(name, start_byte, dtype)
        assert repr(header_spec) == f"TraceHeaderSpec(name='{name}', start_byte={start_byte}, dtype='{dtype}')"

    def test_to_tuple(self, name, start_byte, dtype):
        header_spec = TraceHeaderSpec(name, start_byte, dtype)
        assert header_spec.to_tuple() == (name, start_byte, dtype)
        assert header_spec.to_dict() == {'name': name, 'start_byte': start_byte, 'dtype': dtype}

    def test_hashable(self, name, start_byte, dtype):
        header_spec = TraceHeaderSpec(name, start_byte, dtype)
        set_of_specs = {header_spec}
        assert header_spec in set_of_specs and len(set_of_specs) == 1

@pytest.mark.parametrize('name, start_byte', [('NEW_HEADER', 17)])
@pytest.mark.parametrize('byteorder', [None, '=', '>', '<'])
class TestEndian:
    @pytest.mark.parametrize('dtype', ['f4'])
    def test_default_byteorder(self, name, start_byte, dtype, byteorder):
        header_spec = TraceHeaderSpec(name, start_byte, dtype, byteorder=byteorder)

        assert not header_spec.has_explicit_byteorder

        assert header_spec.has_byteorder == (byteorder is not None)
        assert header_spec.has_default_byteorder == (byteorder is not None)

        assert header_spec.dtype == (byteorder or '') + dtype

    @pytest.mark.parametrize('dtype', ['>f4', '<f4'])
    def test_default_byteorder(self, name, start_byte, dtype, byteorder):
        header_spec = TraceHeaderSpec(name, start_byte, dtype, byteorder=byteorder)

        assert header_spec.has_explicit_byteorder
        assert header_spec.byteorder == dtype[0]


        assert header_spec.has_byteorder
        assert header_spec.has_default_byteorder == (byteorder is not None)

        assert header_spec.dtype == dtype

@pytest.mark.parametrize('name', [None, 'CDP_X', 'NEW_HEADER'])
@pytest.mark.parametrize('start_byte', [None, 181, 182])
@pytest.mark.parametrize('dtype', [None, 'i4', '>i4', '<f8'])
@pytest.mark.parametrize('byteorder', [None, '<', '>'])

class TestMultipleParams:
    def get_expectation(self, name, start_byte, dtype, byteorder):
        if (name is None and start_byte is None or
            name is None and start_byte not in TraceHeaderSpec.STANDARD_BYTE_TO_HEADER or
            start_byte is None and name not in TraceHeaderSpec.STANDARD_HEADER_TO_BYTE or
            start_byte is not None and start_byte not in TraceHeaderSpec.STANDARD_BYTE_TO_HEADER and dtype is None):

            return pytest.raises(KeyError)
        return does_not_raise()

    def test_initialization(self, name, start_byte, dtype, byteorder):
        with self.get_expectation(name, start_byte, dtype, byteorder):
            header_spec = TraceHeaderSpec(name, start_byte, dtype, byteorder)

            assert name is None or header_spec.name == name
            assert start_byte is None or header_spec.start_byte == start_byte
            assert byteorder is None or header_spec.has_default_byteorder and header_spec.default_byteorder == byteorder
            assert dtype is None or header_spec.dtype.str.strip('>').strip('<') == dtype.strip('>').strip('<')
            assert dtype is None or dtype[0] not in ('>', '<') or header_spec.byteorder == dtype[0]
            assert header_spec.byte_len == int(header_spec.dtype.str[-1])

    def test_set_deafult_byteorder(self, name, start_byte, dtype, byteorder):
        with self.get_expectation(name, start_byte, dtype, byteorder):
            header_spec = TraceHeaderSpec(name, start_byte, dtype)
            new_header_spec = header_spec.set_default_byteorder(byteorder)

            assert byteorder is None or new_header_spec.has_default_byteorder and new_header_spec.default_byteorder == byteorder
            assert dtype is None or new_header_spec.dtype.str.strip('>').strip('<') == dtype.strip('>').strip('<')
            assert dtype is None or dtype[0] not in ('>', '<') or new_header_spec.byteorder == dtype[0]

    def test_eq(self, name, start_byte, dtype, byteorder):
        with self.get_expectation(name, start_byte, dtype, byteorder):
            header_spec = TraceHeaderSpec(name, start_byte, dtype, byteorder)
            new_header_spec = TraceHeaderSpec(header_spec.name, header_spec.start_byte, header_spec.dtype, header_spec.byteorder)
            assert header_spec == new_header_spec
