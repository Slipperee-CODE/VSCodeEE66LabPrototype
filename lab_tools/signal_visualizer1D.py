import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

from dataclasses import dataclass
from collections.abc import Callable
from typing import SupportsIndex

from typing import ClassVar

type SignalFunc = Callable[[float],float]

def visualize_signal(axis_func, axis_func_settings, x: SignalFunc, ps: np.ndarray):
    x = np.vectorize(x)
    sampled_x = x(ps)
    axis_func(ps, sampled_x, **axis_func_settings)

def visualize_DT_from_CT_signal(x: SignalFunc, axis_func, axis_func_settings, start_pos: float, end_pos: float, sample_rate: SupportsIndex):
    visualize_signal(axis_func, axis_func_settings, x, np.linspace(start_pos, end_pos, int(int(sample_rate)*(end_pos-start_pos))))

def visualize_CT_signal(x: SignalFunc, axis_func, axis_func_settings, start_pos, end_pos):
    visualize_DT_from_CT_signal(x, axis_func, axis_func_settings, start_pos, end_pos, 1000)

def visualize_DT_signal(x: SignalFunc, axis_func, axis_func_settings, start_pos, end_pos):
    visualize_DT_from_CT_signal(x, axis_func, axis_func_settings, start_pos, end_pos, 1)

class AXES_SETTINGS:
    DEFAULT_STEM: ClassVar[dict] = {"linefmt":'C1-', "markerfmt":'C1o', "basefmt":' ', "label":'DT Signal'}
    DEFAULT_BAR: ClassVar[dict] = {"width":0.6, "color":'#3b82f6', "edgecolor":'black', "linewidth":0.8, "label":'DT Signal'}
    DEFAULT_PLOT: ClassVar[dict] = {"color":'0.75', "lw":1.2, "label":'CT Signal'}

@dataclass
class DTSignal:
    x: SignalFunc
    axis_func_settings: dict
    axis_func: str = "stem"
    x_label: str = "time"
    y_label: str = "amplitude"

@dataclass
class DTSignalFromCTSignal:
    x: SignalFunc
    sample_rate: SupportsIndex
    axis_func_settings: dict
    axis_func: str = "stem"
    x_label: str = "time"
    y_label: str = "amplitude"

@dataclass
class DTSignalFromCTSignalAndCTSignal:
    x: SignalFunc
    sample_rate: SupportsIndex
    axis_func1_settings: dict
    axis_func2_settings: dict
    axis_func1: str = "stem"
    axis_func2: str = "plot"
    x_label: str = "time"
    y_label: str = "amplitude"

@dataclass
class CTSignal:
    x: SignalFunc
    axis_func_settings: dict
    axis_func: str = "plot"
    x_label: str = "time"
    y_label: str = "amplitude"

type Signal = DTSignal | DTSignalFromCTSignal | DTSignalFromCTSignalAndCTSignal | CTSignal

def show_many_signals(xs: list[Signal], start_pos: float, end_pos: float):
    fig, axes = plt.subplots(len(xs), 1, figsize=(9, 6), sharex=True, squeeze=False)
    for x, ax in zip(xs, axes.flat):
        find_and_create_visualization(x, ax, start_pos, end_pos)
    plt.tight_layout()

def show_many_overlapped_signals(xs: list[Signal], start_pos: float, end_pos: float):
    fig, ax = plt.subplots(figsize=(9, 6))
    for x in xs:
        find_and_create_visualization(x, ax, start_pos, end_pos)
    plt.tight_layout()

def find_and_create_visualization(x: Signal, ax: Axes, start_pos: float, end_pos: float):
    match x:
        case DTSignal(x=dt, axis_func_settings=axis_func_settings, axis_func=axis_func, x_label=x_label, y_label=y_label):
            visualize_DT_signal(dt, getattr(ax, axis_func), axis_func_settings, start_pos, end_pos)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        case DTSignalFromCTSignal(x=ct, sample_rate=sample_rate, axis_func_settings=axis_func_settings, axis_func=axis_func, x_label=x_label, y_label=y_label):
            axis_func_settings["label"] = f"DT from CT at {sample_rate} sample_rate"
            visualize_DT_from_CT_signal(ct, getattr(ax, axis_func), axis_func_settings, start_pos, end_pos, sample_rate)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        case DTSignalFromCTSignalAndCTSignal(x=ct, sample_rate=sample_rate, axis_func1_settings=axis_func1_settings, axis_func2_settings=axis_func2_settings, axis_func1=axis_func1, axis_func2=axis_func2, x_label=x_label, y_label=y_label):
            axis_func1_settings["label"] = f"DT from CT at {sample_rate} sample_rate"
            visualize_DT_from_CT_signal(ct, getattr(ax, axis_func1), axis_func1_settings, start_pos, end_pos, sample_rate)
            axis_func2_settings["label"] = f"CT for {sample_rate} sample_rate"
            visualize_CT_signal(ct, getattr(ax, axis_func2), axis_func2_settings, start_pos, end_pos)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        case CTSignal(x=ct, axis_func_settings=axis_func_settings, axis_func=axis_func, x_label=x_label, y_label=y_label):
            visualize_CT_signal(ct, getattr(ax, axis_func), axis_func_settings, start_pos, end_pos)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
    
    ax.legend(loc='upper right')
    
def show_signal(x: Signal, start_pos: float, end_pos: float):
    show_many_signals([x], start_pos, end_pos)