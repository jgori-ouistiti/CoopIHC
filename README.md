![CoopIHC Logo](https://raw.githubusercontent.com/jgori-ouistiti/CoopIHC/main/docs/guide/images/coopihc-logo.png)

_CoopIHC_, pronounced 'kopik', is a Python module that provides a common basis for describing computational Human Computer Interaction (HCI) contexts, mostly targeted at expressing models of users and intelligent assistants.

1. It provides a common conceptual and practical reference, which facilitates reusing and extending other researcher's work
2. It can help design intelligent assistants by translating an interactive context into a problem that can be solved (via other methods).

## Requirements

CoopIHC is known to work with Python 3.8.0 and above.

## Resources

- [Getting Started](https://jgori-ouistiti.github.io/CoopIHC/guide/quickstart.html)
- [Documentation](https://jgori-ouistiti.github.io/CoopIHC/)
- [Contributing](https://github.com/jgori-ouistiti/CoopIHC/blob/main/CONTRIBUTING.md)

## Installing

You can install the package using pip with the following command:

```Shell

pip install coopihc

```

You can them import the necessary packages from `coopihc`.

## Quickstart

At a high level, CoopIHC code will usually consist of defining tasks, agents (users and assistants), bundling them together, and playing several rounds of interaction until the game ends. This generates data that you can then use for something else.


```Python
# Define a task
example_task = ExampleTask()
# Define a user
example_user = ExampleUser()
# Define an assistant
example_assistant = ExampleAssistant()
# Bundle them together
bundle = Bundle(task=example_task, user=example_user, assistant=example_assistant)
# Reset the bundle (i.e. initialize it to a random or prescribed states)
bundle.reset(
    go_to=1
)  # Reset in a state where the user has already produced an observation and made an inference.

# Step through the bundle (i.e. play full rounds)
while 1:
    state, rewards, is_done = bundle.step(user_action=1, assistant_action=None)
    # Do something with the state or the rewards
    if is_done:
        break
```

The point of CoopIHC is to make task and agent definitions generic, to facilitate re-use of various components among researchers. For example, a few user models are released with CoopIHC that can be used off the shelf, and we are actively adding more. Tasks and agents can also be combined freely via a single Bundle object. This allows you to test various user models or assistants with minimal effort, including real user input.



Head over to the [Quickstart](https://jgori-ouistiti.github.io/CoopIHC/guide/quickstart.html) to get a better picture.

## Contribution Guidelines

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

To learn more about making a contribution to CoopIHC, please see our [Contribution page](CONTRIBUTING.md).

Link to the project site: https://jgori-ouistiti.github.io/CoopIHC

## Contributors

Julien Gori (gori@isir.upmc.fr, CNRS, ISIR, Sorbonne Université)
Christoph Johns
