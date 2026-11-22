from lab_tools import make_hint
import ipywidgets as widgets

def HINTS_FOR_make_radial_mask():
    make_hint(widgets.HTML("Try setting the slider to <b>0</b> or <b>100</b> to test the boundary values."), 1)
    make_hint(widgets.HTML("Try <b>NOT</b> setting the slider to <b>0</b> or <b>100</b> to test the boundary values."), 2)
