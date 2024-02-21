""" Utils for interactive selection and validation of trace header specs. """

from functools import partial
from contextlib import contextmanager, ExitStack

import numpy as np

from .memmap_loader import MemmapLoader
from .trace_header_spec import TraceHeaderSpec
from .utils import DelayedImport

# Postpone imports from ipython and ipywidgets as optional
display = DelayedImport("IPython.display", attribute="display")
widgets = DelayedImport("ipywidgets", attribute="widgets")


class Column:
    """ A class defining a column of interactive widgets. The column consists of a title widget and a number of items,
    each created using `construct_item` function on request. Both title and column items may have a callback executed
    on widget value change. """

    def __init__(self, title, construct_item, title_callback=None, item_callback=None):
        self.title = title
        self._title_callback = title_callback
        self.set_callback(self.title, title_callback)

        self._construct_item = construct_item
        self._item_callback = item_callback
        self.item_box = widgets.VBox([])
        self.box = widgets.VBox([self.title, self.item_box])

    @property
    def items(self):
        """ A tuple of column items. """
        return self.item_box.children

    @property
    def n_items(self):
        """ The number of items in the column. """
        return len(self.items)

    @staticmethod
    def set_callback(widget, callback=None):
        """ Set a callback executed on widget value change. Properly handles `Button` widget which has its own callback
        setting logic. """
        if callback is None:
            return
        if isinstance(widget, widgets.Button):
            widget.on_click(callback)
        else:
            widget.observe(callback, names="value")

    @staticmethod
    def reset_callback(widget, callback=None):
        """ Reset a callback executed on widget value change. Properly handles `Button` widget which has its own
        callback resetting logic. """
        if callback is None:
            return
        if isinstance(widget, widgets.Button):
            widget.on_click(callback, remove=True)
        else:
            widget.unobserve(callback, names="value")

    def construct_item(self):
        """ Construct a new column item and set its callback if needed. """
        item = self._construct_item()
        self.set_callback(item, self._item_callback)
        return item

    def append_item(self):
        """ Append a new item to the column. """
        self.item_box.children += (self.construct_item(),)

    def remove_item(self, i):
        """ Remove `i`-th item from the column. """
        if i < 0 or i >= self.n_items:
            return
        self.item_box.children = self.item_box.children[:i] + self.item_box.children[i+1:]

    def get_item_row(self, item):
        """ Get an index of an `item` in the column. """
        for i, col_item in enumerate(self.items):
            if item is col_item:
                return i

    @contextmanager
    def ignore_events(self):
        """ Ignore callback execution upon entering the context manager until exit. """
        self.reset_callback(self.title, self._title_callback)
        for item in self.items:
            self.reset_callback(item, self._item_callback)
        try:
            yield
        finally:
            self.set_callback(self.title, self._title_callback)
            for item in self.items:
                self.set_callback(item, self._item_callback)


class Table:
    """ A class defining a table of interactive widgets, consisting of individual `Column`s. """

    def __init__(self, *columns):
        for col in columns:
            if not isinstance(col, Column):
                raise TypeError
        if any(col.n_items != columns[0].n_items for col in columns[1:]):
            raise ValueError
        self.columns = columns
        self.box = widgets.HBox([col.box for col in columns])

    @property
    def n_rows(self):
        """ The number of rows in the table. """
        return self.columns[0].n_items

    def append_row(self):
        """ Append a new row to the table. """
        for col in self.columns:
            col.append_item()

    def remove_row(self, i):
        """ Remove `i`-th row from the table. """
        if i < 0 or i >= self.n_rows:
            return
        for col in self.columns:
            col.remove_item(i)

    @contextmanager
    def ignore_events(self):
        """ Ignore callback execution upon entering the context manager until exit. """
        with ExitStack() as stack:
            for col in self.columns:
                stack.enter_context(col.ignore_events())
            yield


class TraceHeaderSpecSelector:
    def __init__(self, path, endian="big", headers=None, n_traces=3):
        self.loader = MemmapLoader(path, endian=endian)
        self.file_n_traces = self.loader.n_traces
        self.n_traces = min(n_traces, self.file_n_traces)

        self.dtype_str_to_np = {
            "int8": "i1",
            "int16": "i2",
            "int32": "i4",
            "int64": "i8",
            "uint8": "u1",
            "uint16": "u2",
            "uint32": "u4",
            "uint64": "u8",
            "float16": "f2",
            "float32": "f4",
            "float64": "f8",
        }
        self.dtype_np_to_str = {v: k for k, v in self.dtype_str_to_np.items()}

        WIDGET_HEIGHT = "30px"
        BUTTON_WIDTH = "35px"
        title_layout = widgets.Layout(height=WIDGET_HEIGHT, width="auto", flex="1 1 auto")
        button_layout = widgets.Layout(height=WIDGET_HEIGHT, width=BUTTON_WIDTH)

        # Construct selector table

        self.remove_col = Column(widgets.HTML("", layout=title_layout),
                                 partial(widgets.Button, icon="times", layout=button_layout),
                                 item_callback=self.on_remove)

        name_layout = widgets.Layout(height=WIDGET_HEIGHT, width="auto", min_width="200px", flex="1 1 auto")
        self.name_col = Column(widgets.HTML("Trace header name", layout=title_layout),
                               partial(widgets.Text, continuous_update=False, layout=name_layout),
                               item_callback=self.on_name_change)

        start_byte_layout = widgets.Layout(height=WIDGET_HEIGHT, width="auto", max_width="10ch", flex="1 1 auto")
        self.start_byte_col = Column(widgets.HTML("Start byte", layout=title_layout),
                                     partial(widgets.Text, continuous_update=False, layout=start_byte_layout),
                                     item_callback=self.on_start_byte_change)

        type_layout = widgets.Layout(height=WIDGET_HEIGHT, width="auto", flex="1 1 auto")
        self.type_col = Column(widgets.HTML("Type", layout=title_layout),
                               partial(widgets.Dropdown, value=None, options=list(self.dtype_str_to_np.keys()),
                                       layout=type_layout),
                               item_callback=self.on_selector_change)

        endianness_layout = widgets.Layout(height=WIDGET_HEIGHT, width="auto", flex="1 1 auto")
        self.endianness_col = Column(widgets.HTML("Endianness", layout=title_layout),
                                     partial(widgets.Dropdown, value="=", options=["=", ">", "<"],
                                             layout=endianness_layout),
                                     item_callback=self.on_selector_change)

        self.selector_table = Table(self.remove_col, self.name_col, self.start_byte_col, self.type_col,
                                    self.endianness_col)
        selector_title = widgets.HTML("<center><b>Trace headers loading specification</b></center>",
                                      layout=title_layout)
        self.selector_box = widgets.VBox([selector_title, self.selector_table.box])

        # Construct headers table
        headers_layout = widgets.Layout(height=WIDGET_HEIGHT, width="auto", min_width="100px", border="groove",
                                        flex="1 1 auto")
        headers_cols = [Column(widgets.BoundedIntText(value=i, min=0, max=self.file_n_traces-1, description="Index:",
                                                      style={"description_width": "initial"}, layout=title_layout),
                               partial(widgets.HTML, value="&ensp;-", layout=headers_layout),
                               title_callback=self.on_selector_change)
                        for i in range(n_traces)]
        self.headers_table = Table(*headers_cols)
        headers_title = widgets.HTML("<center><b>Trace indices and their headers</b></center>", layout=title_layout)
        sample_button = widgets.Button(icon="random", layout=button_layout)
        sample_button.on_click(lambda _: self.resample_traces)
        self.headers_box = widgets.VBox([widgets.HBox([headers_title, sample_button]), self.headers_table.box])

        # Construct a box for spec selection
        placeholder = widgets.HTML(layout=button_layout)
        table_box = widgets.HBox([self.selector_box, placeholder, self.headers_box])
        append_row_button = widgets.Button(icon="plus", layout=button_layout)
        append_row_button.on_click(lambda _: self.append_row())
        self.warn_list = []
        self.warn_box = widgets.HTML(layout=widgets.Layout(width="auto"))
        self.selector_box = widgets.VBox([table_box, append_row_button, self.warn_box])

        # Construct a box with file textual headers
        TEXT_HEADER_LINE_LENGTH = 80
        text_header_list = []
        for text_header in self.loader.text:
            text_header = text_header.decode()
            text_header = "<br>".join(text_header[i : i + TEXT_HEADER_LINE_LENGTH]
                                      for i in range(0, len(text_header), TEXT_HEADER_LINE_LENGTH))
            text_header_list.append(text_header)
        text_headers = "<br>".join(text_header_list)
        text_headers = f"<p style='line-height:1.25'> {text_headers} </p>"
        self.text_box = widgets.HTML(text_headers)

        # Construct a box with two tabs: one with spec selectors and another with textual headers
        self.box = widgets.Tab([self.selector_box, self.text_box], titles=["Spec selectors", "Textual headers"],
                               layout=widgets.Layout(width="fit-content"))

        # Initialize the table with passed headers and load them
        self._init_tables(headers)
        display(self.box)

    @property
    def trace_indices(self):
        return np.array([int(col.title.value) for col in self.headers_table.columns])

    @property
    def headers(self):
        return self.get_headers(warn=False)
    
    def get_headers(self, warn=True):
        widget_iter = zip(self.name_col.items, self.start_byte_col.items,
                          self.type_col.items, self.endianness_col.items)
        headers_list = []
        for name, start_byte, dtype, endian in widget_iter:
            name = name.value
            if name == "":
                continue
            if name in {header.name for header in headers_list}:
                if warn:
                    self.warn(f"Header {name} appears more than once, only the first spec is used")
                continue

            start_byte = start_byte.value
            if start_byte == "":
                continue
            try:
                start_byte = int(start_byte)
            except ValueError:
                if warn:
                    self.warn(f"Start byte of header {name} must be a positive integer")
                continue
            if start_byte < 1 or start_byte > 240:
                if warn:
                    self.warn(f"Start byte of header {name} must be between 1 and 240 inclusive")
                continue

            if dtype.value is None:
                continue
            dtype = self.dtype_str_to_np[dtype.value]
            endian = endian.value
            if endian != "=":
                dtype = endian + dtype

            try:
                header = TraceHeaderSpec(name, start_byte, dtype)
            except ValueError:
                if warn:
                    self.warn(f"Header {name} is out of trace headers bounds")
                continue
            header = self.loader.make_headers_specs([header])[0]
            headers_list.append(header)
        return headers_list

    def _init_tables(self, headers=None):
        if headers is None or len(headers) == 0:
            self.append_row()
            return
        headers_list = self.loader.make_headers_specs(headers)
        with self.selector_table.ignore_events():
            for i, header_spec in enumerate(headers_list):
                self.append_row()
                self.name_col.items[i].value = header_spec.name
                self.start_byte_col.items[i].value = str(header_spec.start_byte)
                self.type_col.items[i].value = self.dtype_np_to_str[header_spec.dtype.str[1:]]
        self.reload_headers()

    def append_row(self):
        self.selector_table.append_row()
        self.headers_table.append_row()

    def remove_row(self, i):
        self.selector_table.remove_row(i)
        self.headers_table.remove_row(i)

    def on_remove(self, button):
        i = self.remove_col.get_item_row(button)
        self.remove_row(i)
        self.reload_headers()

    def on_name_change(self, change):
        try:
            header_spec = TraceHeaderSpec(change["new"])
        except:
            self.reload_headers()
            return

        i = self.name_col.get_item_row(change["owner"])
        if self.start_byte_col.items[i].value != "" or self.type_col.items[i].value is not None:
            self.reload_headers()
            return

        with self.selector_table.ignore_events():
            self.start_byte_col.items[i].value = str(header_spec.start_byte)
            self.type_col.items[i].value = self.dtype_np_to_str[header_spec.dtype.str[1:]]
        self.reload_headers()

    def on_start_byte_change(self, change):
        """Autocomplete trace header name and dtype by start byte."""
        try:
            header_spec = TraceHeaderSpec(start_byte=int(change["new"]))
        except:
            self.reload_headers()
            return

        i = self.start_byte_col.get_item_row(change["owner"])
        if self.name_col.items[i].value != "" or self.type_col.items[i].value is not None:
            self.reload_headers()
            return

        with self.selector_table.ignore_events():
            self.name_col.items[i].value = header_spec.name
            self.type_col.items[i].value = self.dtype_np_to_str[header_spec.dtype.str[1:]]
        self.reload_headers()

    def on_selector_change(self, change):
        _ = change
        self.reload_headers()

    def resample_traces(self):
        trace_ix = np.random.randint(self.file_n_traces - 1, size=self.n_traces)
        with self.selector_table.ignore_events():
            for ix, col in zip(trace_ix, self.headers_table.columns):
                col.title.value = ix
        self.reload_headers()

    def update_warn_box(self):
        self.warn_box.value = "<br>".join(self.warn_list)

    def warn(self, warning):
        self.warn_list.append(warning)
        self.update_warn_box()

    def reset_warnings(self):
        self.warn_list = []
        self.update_warn_box()

    def reload_headers(self):
        self.reset_warnings()
        headers = sorted(self.get_headers(warn=True), key=lambda header: header.start_byte)
        if not headers:
            headers_dict = {}
        else:
            res_headers = [headers[0]]
            for header in headers[1:]:
                last_header = res_headers[-1]
                if last_header.start_byte + last_header.byte_len > header.start_byte:
                    self.warn(f"{last_header.name} and {header.name} headers overlap, "
                              f"{header.name} header won't be loaded")
                    continue
                else:
                    res_headers.append(header)
            df = self.loader.load_headers(res_headers, indices=self.trace_indices, reconstruct_tsf=False,
                                          sort_columns=False)
            headers_dict = df.to_dict("list")

        for i, name_widget in enumerate(self.name_col.items):
            values = headers_dict.pop(name_widget.value, ["-"] * self.n_traces)
            for val, col in zip(values, self.headers_table.columns):
                col.items[i].value = "&ensp;" + str(val)


def select_pre_stack_header_specs(path, endian="big", headers=None, n_traces=3):
    if headers is None:
        headers = [
            "FieldRecord", "TraceNumber", "offset", "SourceX", "SourceY", "SourceSurfaceElevation", "SourceUpholeTime",
            "SourceDepth", "GroupX", "GroupY", "ReceiverGroupElevation", "CDP_X", "CDP_Y", "SourceGroupScalar",
            "ElevationScalar", "CDP", "INLINE_3D", "CROSSLINE_3D"
        ]
    return TraceHeaderSpecSelector(path, endian=endian, headers=headers, n_traces=n_traces)


def select_post_stack_header_specs(path, endian="big", headers=None, n_traces=3):
    if headers is None:
        headers = ["TraceNumber", "CDP", "INLINE_3D", "CROSSLINE_3D", "CDP_X", "CDP_Y"]
    return TraceHeaderSpecSelector(path, endian=endian, headers=headers, n_traces=n_traces)
