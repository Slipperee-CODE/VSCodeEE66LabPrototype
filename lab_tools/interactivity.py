import ipywidgets as widgets
from IPython.display import HTML, display

from inspect import signature

from collections.abc import Callable

# SLIDERS

def sliderfy(func: Callable, filled_params: dict):
    func_params = list(signature(func).parameters.keys())
    assert len(func_params) - 1 == len(filled_params), f"slidered({func}, {filled_params}) requires {len(func_params) - 1 - len(filled_params)} more filled params"
    missing_param = list(set(func_params) ^ set(list(filled_params)))[0]
    def new_func(x):
        fully_filled_params = filled_params
        fully_filled_params[missing_param] = x
        return func(**fully_filled_params)
    return new_func

def slidered(func: Callable, filled_params: dict, slider):
    """Sliderfy func then map it to slider. Use it like this: 
    >>> import ipywidgets as widgets
    >>> f = lambda x, y: x + y
    >>> slidered(f, {"y":2}, widgets.IntSlider(value=50, min=0, max=100, description='x: '))
    >>> # Try sliding the slider around!
    """
    widgets.interact(sliderfy(func, filled_params), x=slider)

# HINTS

def make_hint(hint_content: HTML, hint_num: int):
    accordion = widgets.Accordion(children=[hint_content], titles=(f'Show Hint #{hint_num}',))
    accordion.selected_index = None  # Start collapsed
    display(accordion)