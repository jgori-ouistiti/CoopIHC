# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['coopihc',
 'coopihc.agents',
 'coopihc.agents.lqrcontrollers',
 'coopihc.bundle',
 'coopihc.bundle.wrappers',
 'coopihc.examples.basic_examples',
 'coopihc.examples.simple_examples',
 'coopihc.examples.worked_out_examples',
 'coopihc.examples.worked_out_examples.websockets',
 'coopihc.inference',
 'coopihc.interactiontask',
 'coopihc.observation',
 'coopihc.policy',
 'coopihc.space']

package_data = \
{'': ['*']}

install_requires = \
['PyYAML>=6.0,<7.0',
 'gym>=0.17,<0.18',
 'matplotlib>=3,<4',
 'numpy>=1,<2',
 'pytest-timeout>=2.0.2,<3.0.0',
 'scipy>=1.7.3,<2.0.0',
 'stable-baselines3>=1.3.0,<2.0.0',
 'tabulate',
 'websockets>=10.1,<11.0']

setup_kwargs = {
    'name': 'coopihc',
    'version': '0.0.1',
    'description': 'Two-agent component-based interaction environments for computational HCI with Python',
    'long_description': None,
    'author': 'Julien Gori',
    'author_email': None,
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<3.11',
}


setup(**setup_kwargs)
