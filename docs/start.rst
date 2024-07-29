Quick start
===========

* Open the file:

   .. code-block:: python

    import segfast
    segy_file = segfast.open('/path/to/file.sgy')

* Load headers:

   .. code-block:: python

    headers = segy_file.load_headers(['CDP_X', 'CDP_Y', 'INLINE_3D', 'CROSSLINE_3D'])

* Load inline:

   .. code-block:: python

    traces_idx = headers[headers['INLINE_3D'] == INLINE_IDX].index
    inline = segy_file.load_traces(traces_idx)

* Load certain depths from all traces:

   .. code-block:: python

    segy_file.load_depth_slices(DEPTHS)

   The resulting array will have shape ``(n_traces, len(DEPTHS))`` so it must be processed to be transformed
   to an array of the field shape.
