from collections import OrderedDict
import tabulate
import textwrap

class Core:
    def __init__(self):
        super().__init__()


    # Fix this (this could only work for agents)
    @property
    def observation(self):
        return self.inference_engine.buffer[-1]

    @property
    def parameters(self):
        if self.handbook:
            return self.handbook.parameters


    # fix this (untested, used for openai-gym style wrapper)
    @property
    def unwrapped(self):
        return self


class Handbook(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        _str = 'Name: {}\n'.format(self['name'])
        _str += 'Render Modes: {}\nParameters:\n'.format(str(self['render_mode']))
        if self['parameters']:
            print(self['parameters'][0], type(self['parameters'][0]))
            table_header = list(self['parameters'][0].keys())
            rows = []
            for row in self['parameters']:
                rows.append(list([textwrap.fill(v, width = 50) if isinstance(v, str) else v for v in row.values()]))
            _str += tabulate.tabulate(rows, headers = table_header)

        return _str

    @property
    def parameters(self):
        try:
            return self.param_storage
        except AttributeError:
            self.param_storage = {}
            for p in self['parameters']:
                self.param_storage[p['name']] = p['value']
            return self.param_storage
