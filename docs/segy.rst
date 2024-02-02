SEG-Y description
=================

The most complete description can be found in `official SEG-Y specification <https://library.seg.org/pb-assets/technical-standards/seg_y_rev2_0-mar2017-1686080998003.pdf>`_ but here we give
a brief intro into SEG-Y format.
Each SEG-Y file consists of:

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