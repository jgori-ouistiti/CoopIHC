# Questions for Julien

Throughout implementing parameter recovery and modeling recovery, I have encountered the following issues and questions:

- What are the best practices or programming principles, I should follow for implementing my own functions (e.g. regarding class structure, tests, docstrings, typing)?
- What is the logging strategy? What is the difference in purpose between logger.info and print statements?
- What's the difference between spaces and possible values when defining state or actions? --> mostly with discrete spaces for defining "human values"
- How do I suppress all the logs or change its sink? --> not sure, probably needs to be fundamentally restructured (e.g. comment out)
- Which value do state elements assume on reset when no value is supplied? --> randomly sample from possible values (if none supplied, then from space)
- How do I deal with randomness in the task? Is there a way to make the results replicable? --> seeding

# Tasks

- One branch each for:
  - Documenting what I am doing
  - Implementing the model checks
  - Suggesting general changes to the core code
