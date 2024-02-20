SEG-Y description
=================

The most complete description can be found in `the official SEG-Y specification <https://library.seg.org/pb-assets/technical-standards/seg_y_rev2_0-mar2017-1686080998003.pdf>`_ but here we give
a brief intro into SEG-Y format.

The SEG-Y is a binary file divided into several blocks:

- file-wide information block which in most cases takes the first 3600 bytes:

  - **textual header**: the first 3200 bytes are reserved for textual info about the file. Most of the software uses
    this header to keep acquisition meta, date of creation, author, etc.
  - **binary header**: 3200–3600 bytes contain file-wide headers, which describe the number of traces, format used
    for storing numbers, the number of samples for each trace, acquisition parameters, etc.
  - (optional) 3600+ bytes can be used to store the **extended textual information**. If there is such a header,
    then this is indicated by the value in one of the 3200–3600 bytes.

- a sequence of traces, where each trace is a combination of its header and signal data:

  - **trace header** takes the first 240 bytes and describes the meta info about its trace: shot/receiver coordinates,
    the method of acquisition, current trace length, etc. Analogously to binary file header, each trace also
    can have extended headers.
  - **trace data** is usually an array of amplitude values, which can be stored in various numerical types.
    As the original SEG-Y is quite old (1975), one of those numerical formats is IBM float,
    which is very different from standard IEEE floats; therefore, a special caution is required to
    correctly decode values from such files.

For the most part, SEG-Y files are written with constant size of each trace, although the standard itself allows
for variable-sized traces. We do not work with such files.
