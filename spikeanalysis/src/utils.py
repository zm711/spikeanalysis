
import json


def jsonify_parameters(parameters:dict):

    try:
        with open('analysis_parameters.json') as read_file:
            old_params = json.load(read_file)
        old_params.update(parameters)
        new_parameters = old_params
        
    except FileNotFoundError:
        new_parameters = parameters
        

    with open('analysis_parameters.json') as write_file:
        json.dump(new_parameters)