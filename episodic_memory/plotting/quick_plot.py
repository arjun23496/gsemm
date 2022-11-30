from bokeh.plotting import figure, output_file, save
from bokeh.models import (ColumnDataSource,
                          LinearColorMapper,
                          ColorBar,
                          BasicTicker,
                          PrintfTickFormatter)

import numpy as np


def plot_line(x, y=None,
              tools=["wheel_zoom"],
              fig=None,
              **plot_args):
    """
    Plot a line quickly using bokeh. Behaves similar to plt.plot method when it comes to lines

    Parameters
    ----------
    x : list or 1darray
        a list or 1d array of the x values
    y : list or 1darray
        a list or 1d array of the y values
    tools : list of str
        a list of tools supported by bokeh plot
    fig : bokeh figure
        Adds the line plot to the given bokeh figure. If None creates
        another bokeh figure
    plot_args :
    Returns
    -------
    object
        a bokeh figure object
    """
    if y is None:
        y = x
        x = range(len(y))

    default_tools = "pan,reset,save,"
    tools = default_tools + ','.join(tools)
    plot_options = dict(width=600,
                        plot_height=350,
                        tools=tools)

    if fig is None:
        fig = figure(**plot_options)

    line_glyph = fig.line(x, y, **plot_args)
    return fig, line_glyph



def plot_heatmap_from_array(data,
                            tools=["wheel_zoom"],
                            fig=None,
                            **plot_args):
    # extract_data tuples
    data_tuples = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_tuples.append((i, j, data[i, j]))

    return plot_heatmap_from_data_tuples(data_tuples,
                                         tools,
                                         fig=fig,
                                         sq_size=1.0,
                                         **plot_args)


def plot_heatmap_from_data_tuples(data,
                                  tools=["wheel_zoom"],
                                  fig=None,
                                  sq_size=0.05,
                                  **plot_args):
    """
    Creates a heatmap from list of data tuples of the form [(x1, y1, z1), (x2, y2, z2) ...]
    This function works well when the data is spread out

    Parameters
    ----------
    data : list of tuples (<x>, <y>, <z>)
        input data to plot the heatmap of
    tools : list of str
        a list of tools supported by bokeh plot
    fig : bokeh figure
        Adds the line plot to the given bokeh figure. If None creates

    Returns
    -------
    object
        a bokeh figure object
    """
    default_tools = "pan,reset,save,"
    tools = default_tools + ','.join(tools)
    plot_options = dict(width=600,
                        plot_height=350,
                        tools=tools)

    if fig is None:
        fig = figure(**plot_options)

    print("Extracting plot information from data tuples")
    x = [val[0] for val in data]
    y = [val[1] for val in data]
    z = [val[2] for val in data]
    widths = [ sq_size for val in data ]
    heights = [ sq_size for val in data ]

    source = ColumnDataSource(data=dict(x=x, y=y, z=z, w=widths, h=heights))

    # this is the colormap from the original NYTimes plot
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=np.min(z), high=np.max(z))

    print("Plotting...")
    fig.rect(source=source, x='x', y='y', width='w', height='h',
             fill_color={'field': 'z', 'transform': mapper})

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d%%"),
                         label_standoff=6, border_line_color=None)
    fig.add_layout(color_bar, 'right')

    print("Plot returned")
    return fig
