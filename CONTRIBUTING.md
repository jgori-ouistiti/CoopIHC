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

CoopIHC uses poetry to manage dependencies. You can run

```shell
poetry install
```

to install the dependencies. More information on (poetry's website)[https://python-poetry.org/].

Use the --extras flag to install Deep RL dependencies:

```shell
poetry install --extras "rl"
```


When new dependencies are added (even dev dependencies), make sure to update not only the TOML file (`poetry add package`) but also the `requirements.txt`to make sure that the automatic PR checks pass (`poetry export -f requirements.txt --output requirements.txt --dev --without-hashes`).
Finally, run `poetry build` copy the `setup.py` to the root since those will be used by the VMs to install the necessary dependencies.

# Publishing to (Test)PyPI with poetry

```shell
poetry build
poetry export --without-hashes --dev -f requirements.txt --output requirements.txt
poetry run twine upload --repository-url https://test.pypi.org/legacy/ dist/*

```

# Installing locally

It is now possible to install locally via the toml file. For example, to install coopihc in editable mode in coopihczoo, you could add

```shell
coopihc = { path = "../path/to/coopihc", develop = true }
```

to pyproject.toml.

Otherwise, if you need to install the package locally using pip install -e, check (poetry's repo)[https://github.com/python-poetry/poetry/issues/34] and [https://github.com/python-poetry/poetry/discussions/1135]

To get the latest setup.py file, extract the tarball that poetry builds.

# Documentation

Documentation uses [Sphinx](https://www.sphinx-doc.org/en/master/). Make sure it is installed on your side.

Some tips/tricks:

- When you create a new branch, a new branch-specific documentation is also available. For example, https://jgori-ouistiti.github.io/CoopIHC/branch/doc-latex-fix/index.html is the branch specific documentation entry point for a branch named doc-latex-fix. (/ in branch names automatically converted to --)

-

```shell
$ make whtml
```

is equivalent to make html with sphinxoptions -W (warning treated as error)

# Editor settings

Constraints for formatting code include

1. Using black https://pypi.org/project/black/
2. Line wrapping width of 79 (see PEP8 https://www.python.org/dev/peps/pep-0008/)
3. Writing sphinx-style dosctrings with name and extended summary (recommend using a tool such as autoDocString). It should look like that:

```Python
	"""function [summary]

	[extended_summary]

	:param a: [description]
	:type a: [type]
	:param b: [description]
	:type b: [type]
	:return: [description]
	:rtype: [type]
	"""
```

If you are using vsCode, you can paste this into your settings.json

    "python.defaultInterpreterPath": "python3",
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.wordWrap": "wordWrapColumn",
    "editor.wordWrapColumn": 79,
    "autoDocstring.docstringFormat": "sphinx",
    "autoDocstring.includeExtendedSummary": true,
    "autoDocstring.includeName": true

# Performance testing

For performance testing, we are using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) which adds timing to pytest results.
To make use of the benchmarking features, tests need to have a `benchmark` argument which is used to call the test to be benchmarked (see pytest-benchmark documentation for details).
