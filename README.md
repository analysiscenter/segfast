<div align="center">

# SEGFAST

<a href="#installation">Installation</a> • <a href="#benchmarks">Benchmarks</a> • <a href="#getting-started">Getting Started</a>

[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://python.org)
[![Status](https://github.com/analysiscenter/segfast/actions/workflows/status.yml/badge.svg?branch=master&event=push)](https://github.com/analysiscenter/segfast/actions/workflows/status.yml)

</div>

---

**segfast** is a library for interacting with SEG-Y seismic data. Main features are:

* Faster access to read data: both traces headers and values
* Optional bufferization, where user can provide a preallocated memory to load the data into
* Convenient API that relies on `numpy.memmap` for most operations, while providing `segyio` as a fallback engine


## Installation

    # pip / pip3
    pip3 install segfast

    # developer version (add `--depth 1` if needed)
    git clone https://github.com/analysiscenter/segfast.git


## Benchmarks
Timings for reading data along various projections:

|                                |    slide_i |    slide_x |    slide_d |      crop<br/>(256, 256, 500) |    batch<br/>(20, 256, 256, 500)|
|:-------------------------------|-----------:|-----------:|-----------:|------------------------------:|--------------------------------:|
| segyio                         |   2.58254  |   7.16672  | 3041.3     | 941.285                       | 16104.4                         |
| segfast                        |   1.48056  |   3.37418  |   50.1355  |  82.0574                      |  2761.94                        |
| segfast<br/>segyio engine      |   2.92379  |   5.69101  |  225.13    | 117.571                       |  3968.81                        |
| seismiqb                       |   1.46763  |   3.45154  |   50.3333  | 151.877                       |  2738.86                        |
| seismiqb+HDF5                  |   1.04213  |   1.93414  |    1.80567 |  81.3581                      |  2616.83                        |
| segfast <br/>quantized         |   0.252452 |   0.518485 |   56.6672  |   7.71151                     |  1212.74                        |

<!-- ![SlideBenchmarks](https://user-images.githubusercontent.com/26159964/196654661-3ff89a60-c17e-47a5-862f-7f6b814a0df9.png) -->


## Getting started

After installation just import **seismiQB** into your code. A quick demo of our primitives and methods:
```python
import segfast

segfast_file = segfast.open('/path/to/cube.sgy', engine='memmap')     # open file and read some meta info
                                                                      # engine can be `segyio` or `memmap`

segfast_file.load_headers(['INLINE_3D', 'CROSSLINE_3D', ...])         # load requested headers as dataframe

# Data access. All methods support optional buffer as target memory
segfast_file.load_traces([123, 333, 777], buffer=None)                # load traces by their indices
segfast_file.load_depth_slices([111, 222, 1111], buffer=None)         # load depth slices by their indices

segfast_file.convert(format=5)                                        # convert data format to IEEE float32
                                                                      # speeds up operations by a lot

```
You can get more familiar with the library, its functional and timings by reading [examples](examples).
