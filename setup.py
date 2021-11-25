#!/usr/bin/env python

"""CoopIHC: Two-agent component-based interaction environments for computational HCI with Python
CoopIHC, pronounced 'kopik', is a Python module that provides a common basis for
describing computational Human Computer Interaction (HCI) contexts, mostly targeted
at expressing models of users and intelligent assistants.

1. It provides a common conceptual and practical reference, which facilitates reusing
and extending other researcher's work
2. It can help design intelligent assistants by translating an interactive context into
a problem that can be solved (via other methods).
"""

import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

gym = "*"
matplotlib = "*"
tabulate = "*"
scipy = "*"
websockets = "*"
pyyaml = "*"
pandas = "*"
tqdm = "*"
seaborn = "*"
statsmodels = "*"

# This call to setup() does all the work
setup(
    name="coopihc",
    version="0.0.1",
    description="Two-agent component-based interaction environments for computational HCI with Python",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jgori-ouistiti/CoopIHC",
    author="Julien Gori",
    author_email="gori@isir.upmc.fr",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["coopihc"],
    include_package_data=True,
    install_requires=[
        "gym",
        "html2text",
        "matplotlib",
        "scipy",
        "tabulate",
        "websockets",
        # The following are only required because of ModelChecks
        "pandas",
        "tqdm",
        "seaborn",
        "statsmodels",
    ],
)
