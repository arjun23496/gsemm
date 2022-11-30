"""
A convenience file that imports every possible module that may be required

Warnings
-------
Use sparingly
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
import imageio
from PIL import Image

from glob import glob
import json
import os
import sys

import os
import pickle as pkl
import numpy as np
import colorcet as cc
from scipy.special import logsumexp

# plotting tools
from bokeh.io import output_notebook, show, reset_output
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, save
from bokeh.palettes import Inferno, all_palettes, Viridis256
from bokeh.models import (CustomJS, Slider, ColumnDataSource, Legend, 
                         BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, LogColorMapper, 
                          PrintfTickFormatter, Arrow, OpenHead, 
                          NormalHead, VeeHead, Slider)
from bokeh.models import Whisker, HoverTool, Span, ColorBar
from bokeh.transform import linear_cmap, log_cmap, factor_cmap, transform

import copy

from bokeh.core.properties import Instance, String
from bokeh.models import LayoutDOM
from bokeh.util.compiler import TypeScript

import numpy as np
from tqdm import tqdm, trange

import importlib
