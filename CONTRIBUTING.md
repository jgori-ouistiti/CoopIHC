# Contributing to CoopIHC

Thank you for your interest in contributing to CoopIHC! Before you begin writing code, it is important
that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
   - Post about your intended feature in an [issue](https://github.com/jgori-ouistiti/interaction-agents/issues),
     and we shall discuss the design and implementation. Once we agree that the plan looks good,
     go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
   - Search for your issue in the [CoopIHC issue list](https://github.com/jgori-ouistiti/interaction-agents/issues).
   - Pick an issue and comment that you'd like to work on the feature or bug-fix.
   - If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, please submit a Pull Request to
https://github.com/jgori-ouistiti/interaction-agents.


# Dependencies 

CoopIHC uses pipenv to manage dependencies. You should run 

```Shell

pipenv install

```
to get the required dependencies.
CoopIHC does not ship a requirements.txt file, but you can recreate it from the Pipfile with

```Shell

pipenv lock -r > requirements.txt

```

if you want to use that mechanism.


Some examples require matplotlib, which itself requires a graphical backend to display graphs. Since many backends exist, CoopIHC does not require a specific one as a dependency. If the examples do not display, then you should likely either install a graphical backend (e.g. pyqt5, see matplotlib documentation for others) with pipenv, or, if you have a system-wide install for some backend (likely) go to the configuration file of your pipfile and allow system-wide packages to be installed.

# Editor settings

Constraints for formatting code include 
1. Using black https://pypi.org/project/black/
2. Line wrapping width of 79 (see PEP8 https://www.python.org/dev/peps/pep-0008/)

If you are using vsCode, you can paste this into your settings.json 

    "python.defaultInterpreterPath": "python3",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.wordWrap": "wordWrapColumn",
    "editor.wordWrapColumn": 79
