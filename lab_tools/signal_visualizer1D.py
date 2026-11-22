import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np

from dataclasses import dataclass
from collections.abc import Callable
from typing import SupportsIndex

from typing import ClassVar

type SignalFunc = Callable[[float],float]

def visualize_signal(axis_func, x: SignalFunc, ps: np.ndarray):
    x = np.vectorize(x)
    sampled_x = x(ps)
    axis_func(ps, sampled_x)

def create_plot_func(axis: Axes, plot_settings: dict):
    def plot(xs, ys):
        axis.plot(xs, ys, **plot_settings)
    return plot

def create_stem_func(axis: Axes, stem_settings: dict):
    def stem(xs, ys):
        axis.stem(xs, ys, **stem_settings)
    return stem

def visualize_DT_from_CT_signal(x: SignalFunc, axis_func, start_pos: float, end_pos: float, sample_rate: SupportsIndex):
    visualize_signal(axis_func, x, np.linspace(start_pos, end_pos, int(int(sample_rate)*(end_pos-start_pos))))

def visualize_CT_signal(x: SignalFunc, axis_func, start_pos, end_pos):
    visualize_DT_from_CT_signal(x, axis_func, start_pos, end_pos, 1000)

def visualize_DT_signal(x: SignalFunc, axis_func, start_pos, end_pos):
    visualize_DT_from_CT_signal(x, axis_func, start_pos, end_pos, 1)

@dataclass
class DTSignal:
    x: SignalFunc
    stem_settings: dict
    DEFAULT_STEM: ClassVar[dict] = {"linefmt":'C1-', "markerfmt":'C1o', "basefmt":' ', "label":'DT Signal'}
    x_label: str = "time"
    y_label: str = "amplitude"

@dataclass
class DTSignalFromCTSignal:
    x: SignalFunc
    stem_settings: dict
    sample_rate: SupportsIndex
    x_label: str = "time"
    y_label: str = "amplitude"

@dataclass
class DTSignalFromCTSignalAndCTSignal:
    x: SignalFunc
    stem_settings: dict
    plot_settings: dict
    sample_rate: SupportsIndex
    x_label: str = "time"
    y_label: str = "amplitude"

@dataclass
class CTSignal:
    x: SignalFunc
    plot_settings: dict
    DEFAULT_PLOT: ClassVar[dict] = {"color":'0.75', "lw":1.2, "label":'CT Signal'}
    x_label: str = "time"
    y_label: str = "amplitude"

type Signal = DTSignal | DTSignalFromCTSignal | DTSignalFromCTSignalAndCTSignal | CTSignal

def show_overlapped_signals(xs: list[Signal], start_pos: float, end_pos: float):
    fig, axes = plt.subplots(len(xs), 1, figsize=(9, 6), sharex=True, squeeze=False)
    for x, ax in zip(xs, axes.flat):
        match x:
            case DTSignal(x=dt, x_label=x_label, y_label=y_label, stem_settings=stem_settings):
                visualize_DT_signal(dt, create_stem_func(ax, stem_settings), start_pos, end_pos)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

            case DTSignalFromCTSignal(x=ct, x_label=x_label, y_label=y_label, stem_settings=stem_settings, sample_rate=sample_rate):
                stem_settings["label"] = f"DT from CT at {sample_rate} sample_rate"
                visualize_DT_from_CT_signal(ct, create_stem_func(ax, stem_settings), start_pos, end_pos, sample_rate)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

            case DTSignalFromCTSignalAndCTSignal(x=ct, x_label=x_label, y_label=y_label, stem_settings=stem_settings, plot_settings=plot_settings, sample_rate=sample_rate):
                stem_settings["label"] = f"DT from CT at {sample_rate} sample_rate"
                visualize_DT_from_CT_signal(ct, create_stem_func(ax, stem_settings), start_pos, end_pos, sample_rate)
                plot_settings["label"] = f"CT for {sample_rate} sample_rate"
                visualize_CT_signal(ct, create_plot_func(ax, plot_settings), start_pos, end_pos)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

            case CTSignal(x=ct, x_label=x_label, y_label=y_label, plot_settings=plot_settings):
                visualize_CT_signal(ct, create_plot_func(ax, plot_settings), start_pos, end_pos)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
        
        ax.legend(loc='upper right')
    plt.tight_layout()

def show_signal(x: Signal, start_pos: float, end_pos: float):
    show_overlapped_signals([x], start_pos, end_pos)