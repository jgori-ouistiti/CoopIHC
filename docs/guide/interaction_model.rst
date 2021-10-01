.. interaction_model:

The Interaction Model
======================
*interaction-agents* builds on a two-agent interaction model called the **user-assistant model**, which is represented in the figure below.


.. _interaction_model_fig_label:

.. figure::  images/bundle.png
    :width: 100%

    The graphical interaction model used in *interaction-agents*

The assumptions of the model are:

1. There is a task which can be represented by a state :math:`s_T`,
2. An user wants to drive the task to a goal state,
3. An assistant is here to help (assist) the user achieve this.

In the HCI language, the user is the user and the assistant is some intelligent tool/interface.
Many interactive settings can be decomposed like this, see :ref:`modules-label` for some examples that have been implemented using *interaction-agents*.


Agents
-----------
Both the user and assistant are **agents**:

1. They have an internal state (:math:`s_O` and :math:`s_A` for respectively the user and the assistant), which stores e.g. goals, preferences, model parameters.
2. They have the ability to observe (perfectly or partially) the various states of the components of the interaction model. When making an observation (:math:`o_O` and :math:`o_A`), the agents may receive a reward, to account for the fact that they could perceive a cost (e.g. because that observation may take some time to be created) or a benefit (e.g. because it satisfies a curiosity) to making an observation.
3. Based on these observations, they are able to make inferences that change their internal states. The agents may again here receive a reward, since there could be a cost to inferring (e.g. mental effort, computational resources) or a benefit (e.g. satisfaction).
4. Based on their internal states and their observations, they are able to take actions (:math:`a_O` and :math:`a_A`) via a policy. Those actions may have an effect on the other states (e.g. the task state) of the interaction model.


State Transitions and Observations
--------------------------------------
It is further assumed in the model that the agents act sequentially. A **round** of interaction consists of the following transitions, where the superscript indicates the **turn** of interaction (a round is played in four turns):

.. math::

    s^{(0)} \rightarrow o'_O \rightarrow s^{(1)} \rightarrow a'_O \rightarrow s^{(2)} \rightarrow o'_A \rightarrow s^{(3)} \rightarrow a'_A \rightarrow s^{'(0)}.

Each turn corresponds to specific behaviors of either one of the agents, see the figure below

.. tikz:: Order of Sequential Operations
    :include: tikz/sequential_operations.tikz
    :xscale: 50
    :align: center


For different turns, we define the following joint states

.. math::

    s^{(0)} = (0, s_T, s_O, s_A), \\
    s^{(1)} = (1, s_T, s'_O, s_A), \\
    s^{(2)} = (2, s'_T, s'_O, s_A), \\
    s^{(3)} = (3, s'_T, s'_O, s'_A), \\
    s^{'(0)} = (0, s''_T, s'_O, s'_A),


the joint observations

.. math::

    o^{(1)} = (o'_O, \text{No-Op}),\\
    o^{(2)} = (\text{No-Op}, \text{No-Op}),\\
    o^{(3)} = (\text{No-Op}, o'_A),\\
    o^{'(0)} = (\text{No-Op}, \text{No-Op}),

and the joint actions:

.. math::

    a^{(0)} = (\text{No-Op}, \text{No-Op}),\\
    a^{(1)} = (a'_O, \text{No-Op}),\\
    a^{(2)} = (\text{No-Op}, \text{No-Op}),\\
    a^{(3)} =  (\text{No-Op}, a'_A).


.. note::

    This sequential nature of actions is not a strong assumption, since simultaneous play can always be achieved by delaying the effect of the user action to the assistant's turn. Furthermore, it could be argued that no real simultaneous play is ever possible.

.. note::

    No-Op is a No-Operation, and signals that this part of the joint action is void.

A Partially Observable Stochastic Game (POSG)
-----------------------------------------------

The user-assistant model is a Partially Observable Stochastic Game (POSG), with transition and observation probabilities:

.. math::

    p(s^{(1)}, o^{(1)} | s^{(0)}, a^{(0)}) & = p(o^{(1)} | s^{(0)}, a^{(0)}) \cdot{} p(s^{(1)}| o^{(1)}, s^{(0)}, a^{(0)}) \\
    &= \underbrace{p(o'_O | s^{(0)})}_\text{user observation function      }  \underbrace{p(s^{(1)}| o'_O, s^{(0)})}_\text{        user inference function} \\
    p(s^{(2)}, o^{(2)} | s^{(1)}, a^{(1)}) & = p(s^{(2)}| s^{(1)}, a^{(1)}) \\
    &= \underbrace{p(s^{(2)}| s^{(1)}, a'_O)}_\text{        user step function} \\
    p(s^{(3)}, o^{(3)} | s^{(2)}, a^{(2)}) & = p(o^{(3)} | s^{(2)}, a^{(2)}) \cdot{} p(s^{(3)}| o^{(3)}, s^{(2)}, a^{(2)}) \\
    &= \underbrace{p(o'_A | s^{(2)})}_\text{assistant observation function      }  \underbrace{p(s^{(3)}| o'_A, s^{(2)})}_\text{        assistant inference function} \\
    p(s^{'(0)}, o^{'(0)} | s^{(3)}, a^{(3)}) & = p(s^{'(0)}| s^{(3)}, a^{(3)}) \\
    &= \underbrace{p(s^{'(0)}| s^{(3)}, a'_A)}_\text{        assistant step function} \\


Hence, from now on we will talk of an **interactive game** (or simply a **game**) when discussing this interactive setting.



Bundles
---------------------
The interactive game is created from the three components (task, user and assistants) using so called **bundles**. Bundles allow the synchronization of all the states of the various components and produce the game state :math:`s = (k, s_T, s_O, s_A)`.


.. note::

    Bundles also handle joint rendering, as well as collection of rewards.


Several bundles exist, to form various games with different utilities for the user of *interaction-agents*. Current implemented bundles are:

1. ``PlayNone`` , which does not take any action as input. It puts together two agents together at play to perform the task. This allows one to evaluate agents where policies are provided (e.g. a trained agent, a rule-based agent).
2. ``PlayUser`` , which takes an user action as input at each step. It puts together an assistant with a defined policy, and an user without policy. This allows one to evaluate a policy on line (e.g. as part of a training procedure).
3. ``PlayAssistant``, which is the counterpart to the previous bundle with users and assistants switched.
4. ``PlayBoth``, which takes the joint (user, assistant) action as input at each step. This allows evaluating policies for both agents on-line (e.g. as part of a joint training procedure).
5. ``SinglePlayUser``, which does not uses an assistant. It it useful when one wants to develop a "pure" user model using *interaction-agents*. Here the policy is evaluated on-line.
6. ``SinglePlayUserAuto`` is the same as the previous bundle but the policy is assumed to be provided to the agent.
7. ``_DevelopTask`` is a bundle with only the task as input, used to develop tasks when no compatible agents exist yet.
