
import json
from typing import Union
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def jsonify_parameters(parameters:dict):

    try:
        with open('analysis_parameters.json') as read_file:
            old_params = json.load(read_file)
        old_params.update(parameters)
        new_parameters = old_params
        
    except FileNotFoundError:
        new_parameters = parameters
        

    with open('analysis_parameters.json') as write_file:
        json.dump(new_parameters, write_file)



def verify_window_format(window: Union[list,list[list]], num_stim: int)->list[list]:

    """Utility function for making sure window format is correct for analysis 
    and plotting functions

    Parameters
    ----------
    window: Union[list, list[list]]
        Either a single sequence of (start, stop) or one list per stimulus
        each containing their own (start, stop)
    num_stim: int
        The number of stimuli being analyzed
        
    Returns
    -------
    windows: list[list]
        A list of lists with each stimulus having a sequence of start stop"""

    if len(window) == 2 and isinstance(window[0], (int, float)):
        windows = [window] * num_stim
    else:
        assert (
            len(window) == num_stim
        ), f"Please enter correct number of lists for stim \
            bsl_window length is {len(window)} and should be {num_stim}"
        windows= window

    return windows