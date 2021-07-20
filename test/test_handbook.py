from core.core import Handbook
import sys
_str = sys.argv[1]

# -------- Correct assigment
if _str == 'str' or _str == 'all':

    handbook = Handbook({'name': 'name', 'render_mode': [], 'parameters': []})
    handbook['render_mode'].extend(['plot', 'text'])
    _gridsize = {"name": 'gridsize', "value": 1, "meaning": 'Size of the gridworld'}
    _number_of_targets = { "name": 'number_of_targets', "value": 1, 'meaning': 'number of potential targets from which to choose a goal'}
    _mode = { "name": 'mode','value': 'yes', 'meaning': "whether the assistant is expected to work as gain or as positioner. In the first case (gain), the operator's action is multiplied by the assistant's action to determine by how much to shift the old position of the cursor. In the second case (position) the assistant's action is directly the new position of the cursor."}
    handbook['parameters'].extend([_gridsize, _number_of_targets, _mode])

    print(handbook)

if _str == 'paramprop' or _str == 'all':

    handbook = Handbook({'name': 'name', 'render_mode': [], 'parameters': []})
    handbook['render_mode'].extend(['plot', 'text'])
    _gridsize = {"name": 'gridsize', "value": 1, "meaning": 'Size of the gridworld'}
    _number_of_targets = { "name": 'number_of_targets', "value": 1, 'meaning': 'number of potential targets from which to choose a goal'}
    _mode = { "name": 'mode','value': 'yes', 'meaning': "whether the assistant is expected to work as gain or as positioner. In the first case (gain), the operator's action is multiplied by the assistant's action to determine by how much to shift the old position of the cursor. In the second case (position) the assistant's action is directly the new position of the cursor."}
    handbook['parameters'].extend([_gridsize, _number_of_targets, _mode])

    print(handbook.parameters)
