.. segfast documentation master file, created by
   sphinx-quickstart on Thu Feb  1 14:09:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

segfast documentation
=====================

**segfast** is a library for interacting with SEG-Y seismic data. Main features are:

* Faster access to read data: both traces headers and values
* Optional bufferization, where user can provide a preallocated memory to load the data into
* Convenient API that relies on :class:`numpy.memmap` for most operations, while providing
  `segyio <https://segyio.readthedocs.io/en/latest/>`_ as a fallback engine


Implementation details
----------------------
We rely on **segyio** to infer file-wide parameters.

For headers and traces, we use custom methods of reading binary data.

Main differences to **segyio** C++ implementation:
   - we read all of the requested headers in one file-wide sweep, speeding up by an order of magnitude
     compared to the **segyio** sequential read of every requested header.
     Also, we do that in multiple processes across chunks.

   - a memory map over traces data is used for loading values. Avoiding redundant copies and leveraging
     :mod:`numpy` superiority allows to speed up reading, especially in case of trace slicing along the samples axis.
     This is extra relevant in case of loading horizontal (depth) slices.


.. toctree::
   :maxdepth: 1
   :titlesonly:

   installation
   start
   segy
   api/segfast
