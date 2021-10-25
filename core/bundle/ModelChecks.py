# Core libraries
from core.bundle import Bundle
from core.helpers import (
    order_class_parameters_by_signature,
    bic,
    aic,
    f1,
)

# Standard libraries
from dataclasses import dataclass
import pandas as pd
import matplotlib.axes
import numpy
from tqdm import tqdm
import inspect
from copy import copy
import sys
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import statsmodels.stats.proportion
from tabulate import tabulate
from collections import OrderedDict


class ModelChecks(Bundle):
    """A bundle without an assistant. It can be used when developing users and
    includes methods for modeling checks (e.g. parameter or model recovery).

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) An user, which is a subclass of BaseAgent
    :param kwargs: Additional controls to account for some specific subcases, see Doc for a full list
    """

    def _user_can_compute_likelihood(user):
        """Returns whether the specified user's policy has a method called "compute_likelihood".

        :param user: An user, which is a subclass of BaseAgent
        :type user: core.agents.BaseAgent
        """
        # Method name
        COMPUTE_LIKELIHOOD = "compute_likelihood"

        # Method exists
        policy_has_attribute_compute_likelihood = hasattr(
            user.policy, COMPUTE_LIKELIHOOD
        )

        # Method is callable
        compute_likelihood_is_a_function = callable(
            getattr(user.policy, COMPUTE_LIKELIHOOD)
        )

        # Return that both exists and is callable
        user_can_compute_likelihood = (
            policy_has_attribute_compute_likelihood and compute_likelihood_is_a_function
        )
        return user_can_compute_likelihood

    @dataclass
    class ParameterRecoveryTestResult:
        """Represents the results of a test for parameter recovery."""

        correlation_data: pd.DataFrame
        """The 'true' and recovered parameter value pairs"""

        correlation_statistics: pd.DataFrame
        """The correlation statistics (i.e. the correlation coefficient and its p-value) for the parameter recovery"""

        parameters_can_be_recovered: bool
        """`True` if the correlation between used and recovered parameter values meets the supplied thresholds, `False` otherwise"""

        recovered_parameters_correlate: bool
        """`True` if any correlation between two recovered parameters exceeds the supplied threshold, `False` otherwise"""

        plot: matplotlib.axes.Axes
        """The scatterplot displaying the 'true' and recovered parameter values for each parameter"""

        correlation_threshold: float
        """The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered parameters)"""

        significance_level: float
        """The threshold for the p-value to consider the correlation significant"""

        n_simulations: int
        """The number of agents that were simulated (i.e. the population size) for the parameter recovery"""

        @property
        def success(self):
            """`True` if all parameters can be recovered and none of the recovered parameters correlate, `False` otherwise"""
            return (
                self.parameters_can_be_recovered and self.recovered_parameters_correlate
            )

    def test_parameter_recovery(
        self,
        parameter_fit_bounds,
        correlation_threshold=0.7,
        significance_level=0.05,
        n_simulations=20,
        n_recovery_trials_per_simulation=1,
        recovered_parameter_correlation_threshold=0.5,
        seed=None,
        workflow="maximum-likelihood",
        **kwargs,
    ):
        """Returns whether the recovered user parameters correlate to the used parameters for a simulation given the supplied thresholds
        and that the recovered parameters do not correlate (test only available for users with a policy that has a compute_likelihood method).

        It simulates n_simulations agents of the user's class using random parameters within the supplied parameter_fit_bounds,
        executes the provided task and tries to recover the user's parameters from the simulated data. These recovered parameters are then
        correlated to the originally used parameters for the simulation using Pearson's r and checks for the given correlation and significance
        thresholds.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param correlation_threshold: The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered
            parameters), defaults to 0.7
        :type correlation_threshold: float, optional
        :param significance_level: The threshold for the p-value to consider the correlation significant, defaults to 0.05
        :type significance_level: float, optional
        :param n_simulations: The number of agents to simulate (i.e. the population size) for the parameter recovery, defaults to 20
        :type n_simulations: int, optional
        :param n_recovery_trials_per_simulation: The number of trials to recover the true parameter value (i.e. to determine the
            best-fit parameter values) for one set of simulated data, defaults to 1
        :type n_recovery_trials_per_simulation: int, optional
        :param recovered_parameter_correlation_threshold: The threshold for Pearson's r value between the recovered parameters, defaults to 0.7
        :type recovered_parameter_correlation_threshold: float, optional
        :param seed: The seed for the random number generator which controls how the 'true' parameter values are generated, defaults to None
        :type seed: int, optional
        :param workflow: The kind of inference technique or workflow that will be used (one of "maximum-likelihood"
            or "bayesian"), defaults to "maximum-likelihood"
        :type workflow: str, optional
        :return: The result of the parameter recovery test
        :rtype: ModelChecks.ParameterRecoveryResult
        """
        # Transform the specified dict of parameter fit bounds into an OrderedDict
        # based on the order of parameters in the user class constructor
        ordered_parameter_fit_bounds = order_class_parameters_by_signature(
            self.user.__class__, parameter_fit_bounds
        )

        # Depending on specified workflow...
        if workflow == "maximum-likelihood":
            return self.test_mle_parameter_recovery(
                parameter_fit_bounds=parameter_fit_bounds,
                correlation_threshold=correlation_threshold,
                significance_level=significance_level,
                n_simulations=n_simulations,
                n_recovery_trials_per_simulation=n_recovery_trials_per_simulation,
                recovered_parameter_correlation_threshold=recovered_parameter_correlation_threshold,
                seed=seed,
                **kwargs,
            )

        elif workflow == "bayesian":
            return self.test_bayesian_parameter_recovery()

        else:
            raise ValueError(
                "Sorry, no other workflow has been implemented yet. workflow needs to be either 'maximum-likelihood' or 'bayesian'"
            )

    @dataclass
    class BayesianParameterRecoveryTestResult:
        """Represents the results of a test for parameter recovery using a Bayesian approach."""

        success: bool = False
        """`True` if all parameters can be recovered and none of the recovered parameters correlate, `False` otherwise"""

    def test_bayesian_parameter_recovery(self):
        """Returns whether the recovered user parameters correlate to the used parameters for a simulation given the supplied thresholds
        and that the recovered parameters do not correlate using a Bayesian approach (test only available for users with a policy
        that has a compute_likelihood method).
        """
        return ModelChecks.BayesianParameterRecoveryTestResult(success=True)

    def test_mle_parameter_recovery(
        self,
        parameter_fit_bounds,
        correlation_threshold=0.7,
        significance_level=0.05,
        n_simulations=20,
        n_recovery_trials_per_simulation=1,
        recovered_parameter_correlation_threshold=0.5,
        seed=None,
        **kwargs,
    ):
        """Returns whether the recovered user parameters correlate to the used parameters for a simulation given the supplied thresholds
        and that the recovered parameters do not correlate using a maximum-likelihood approach (test only available for users with a policy
        that has a compute_likelihood method).

        It simulates n_simulations agents of the user's class using random parameters within the supplied parameter_fit_bounds,
        executes the provided task and tries to recover the user's parameters from the simulated data. These recovered parameters are then
        correlated to the originally used parameters for the simulation using Pearson's r and checks for the given correlation and significance
        thresholds.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param correlation_threshold: The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered
            parameters), defaults to 0.7
        :type correlation_threshold: float, optional
        :param significance_level: The threshold for the p-value to consider the correlation significant, defaults to 0.05
        :type significance_level: float, optional
        :param n_simulations: The number of agents to simulate (i.e. the population size) for the parameter recovery, defaults to 20
        :type n_simulations: int, optional
        :param n_recovery_trials_per_simulation: The number of trials to recover the true parameter value (i.e. to determine the
            best-fit parameter values) for one set of simulated data, defaults to 1
        :type n_recovery_trials_per_simulation: int, optional
        :param recovered_parameter_correlation_threshold: The threshold for Pearson's r value between the recovered parameters, defaults to 0.7
        :type recovered_parameter_correlation_threshold: float, optional
        :param seed: The seed for the random number generator which controls how the 'true' parameter values are generated, defaults to None
        :type seed: int, optional
        :param workflow: The kind of inference technique or workflow that will be used (one of "maximum-likelihood"
            or "bayesian"), defaults to "maximum-likelihood"
        :type workflow: str, optional
        :return: The result of the parameter recovery test
        :rtype: ModelChecks.ParameterRecoveryResult
        """
        # Compute the likelihood data (i.e. the used and recovered parameter pairs)
        correlation_data = self._likelihood(
            parameter_fit_bounds=parameter_fit_bounds,
            n_simulations=n_simulations,
            n_recovery_trials_per_simulation=n_recovery_trials_per_simulation,
            seed=seed,
            **kwargs,
        )

        # Plot the correlations between the used and recovered parameters as a graph
        regplot = ModelChecks._correlations_plot(
            parameter_fit_bounds=parameter_fit_bounds,
            data=correlation_data,
            kind="reg",
        )

        # Compute the correlation metric Pearson's r and its significance for each parameter pair and return it
        correlation_statistics = self._pearsons_r(
            parameter_fit_bounds=parameter_fit_bounds,
            data=correlation_data,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
        )

        # Check that all correlations meet the threshold and are significant
        parameters_can_be_recovered = ModelChecks._correlations_meet_thresholds(
            correlation_statistics=correlation_statistics,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
        )

        # Test whether recovered parameters correlate
        parameter_count = len(parameter_fit_bounds)
        recovered_parameters_correlate = (
            ModelChecks._recovered_parameters_correlate(
                data=correlation_data,
                correlation_threshold=recovered_parameter_correlation_threshold,
            )
            if parameter_count > 1
            else False
        )

        # Create result object and return it
        result = ModelChecks.ParameterRecoveryTestResult(
            correlation_data=correlation_data,
            correlation_statistics=correlation_statistics,
            parameters_can_be_recovered=parameters_can_be_recovered,
            recovered_parameters_correlate=recovered_parameters_correlate,
            plot=regplot,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
            n_simulations=n_simulations,
        )
        return result

    def _recovered_parameters_correlate(data, correlation_threshold=0.7):
        """Returns whether the provided recovered parameters correlate with the specified threshold.

        :param data: A DataFrame containing the subject identifier, parameter name and recovered parameter value
        :type data: pandas.DataFrame
        :param correlation_threshold: The threshold for a parameter pair to be considered correlated, defaults to 0.7
        :type correlation_threshold: float, optional
        :return: `True` if any of the recovered parameters correlate meeting the specified threshold
        :rtype: bool
        """
        # Test whether recovered parameters correlate
        # Pivot data so that each parameter has its own column
        pivoted_correlation_data = data.pivot(
            index="Subject", columns="Parameter", values="Recovered"
        )

        # Calculate correlation matrix
        correlation_matrix = pivoted_correlation_data.corr()

        # Delete duplicate and diagonal values
        correlation_matrix = correlation_matrix.mask(
            numpy.tril(numpy.ones(correlation_matrix.shape)).astype(numpy.bool)
        )

        # Transform data so that each pair-wise correlation that meets threshold is one row
        correlations = correlation_matrix.stack()

        # Select only those correlations that pass specified threshold
        strong_correlations = correlations.loc[
            abs(correlations) > correlation_threshold
        ]

        # Determine whether 'strong' correlations exist between the recovered parameters
        recovered_parameters_correlate = len(strong_correlations) > 0

        # Return `True` if the recovered parameter values are correlated
        return recovered_parameters_correlate

    def _correlations_meet_thresholds(
        correlation_statistics, correlation_threshold=0.7, significance_level=0.05
    ):
        """Returns `True` if all correlation coefficients for the parameter recovery meet the required thresholds.

        :param correlation_statistics: The correlation statistics (i.e. the correlation coefficient and its p-value) for the parameter recovery
        :type correlation_statistics: pandas.DataFrame
        :param correlation_threshold: The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered
            parameters), defaults to 0.7
        :type correlation_threshold: float, optional
        :param significance_level: The threshold for the p-value to consider the correlation significant, defaults to 0.05
        :type significance_level: float, optional
        :return: Returns `True` if all correlation coefficients for the parameter recovery meet the required thresholds, `False` otherwise
        :rtype: bool
        """
        # Check that all correlations meet the threshold and are significant and return the result
        all_correlations_meet_threshold = correlation_statistics[
            f"r>{correlation_threshold}"
        ].all()
        all_correlations_significant = correlation_statistics[
            f"p<{significance_level}"
        ].all()

        return all_correlations_meet_threshold and all_correlations_significant

    def _likelihood(
        self,
        parameter_fit_bounds,
        n_simulations=20,
        n_recovery_trials_per_simulation=1,
        seed=None,
        **kwargs,
    ):
        """Returns a DataFrame containing the likelihood of each recovered parameter.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param n_simulations: The number of agents to simulate (i.e. the population size) for the parameter recovery, defaults to 20
        :type n_simulations: int, optional
        :param n_recovery_trials_per_simulation: The number of trials to recover the true parameter value (i.e. to determine the
            best-fit parameter values) for one set of simulated data, defaults to 1
        :type n_recovery_trials_per_simulation: int, optional
        :param seed: The seed for the random number generator which controls how the 'true' parameter values are generated, defaults to None
        :type seed: int, optional
        :return: A DataFrame containing the likelihood of each recovered parameter.
        :rtype: pandas.DataFrame
        """
        # Make sure user has a policy that can compute likelihood of an action given an observation
        user_can_compute_likelihood = ModelChecks._user_can_compute_likelihood(
            self.user
        )

        # If it cannot compute likelihood...
        if not user_can_compute_likelihood:
            # Raise an exception
            raise ValueError(
                "Sorry, the given checks are only implemented for user's with a policy that has a compute_likelihood method so far."
            )

        # Data container
        likelihood_data = []

        # Random number generator
        random_number_generator = numpy.random.default_rng(seed)

        # For each agent...
        for i in tqdm(range(n_simulations), file=sys.stdout):

            # Generate a random agent
            attributes_from_self_user = {
                attr: getattr(self.user, attr)
                for attr in dir(self.user)
                if attr in inspect.signature(self.user.__class__).parameters
            }
            random_agent = None
            if not len(parameter_fit_bounds) > 0:
                random_agent = self.user.__class__(**attributes_from_self_user)
            else:
                random_parameters = ModelChecks._random_parameters(
                    parameter_fit_bounds=parameter_fit_bounds,
                    random_number_generator=random_number_generator,
                )
                random_agent = self.user.__class__(
                    **{**attributes_from_self_user, **random_parameters}
                )

            # Simulate the task
            simulated_data = self._simulate(
                user=random_agent, random_number_generator=random_number_generator
            )

            # For n_recovery_trials_per_simulation...
            for _ in range(n_recovery_trials_per_simulation):

                # Determine best-fit parameter values
                best_fit_parameters, _ = ModelChecks.best_fit_parameters(
                    task=self.task,
                    user_class=self.user.__class__,
                    parameter_fit_bounds=parameter_fit_bounds,
                    data=simulated_data,
                    random_number_generator=random_number_generator,
                    **kwargs,
                )

                # Backup parameter values
                for parameter_index, (parameter_name, parameter_value) in enumerate(
                    random_parameters.items()
                ):
                    _, best_fit_parameter_value = best_fit_parameters[parameter_index]
                    likelihood_data.append(
                        {
                            "Subject": i + 1,
                            "Parameter": parameter_name,
                            "Used to simulate": parameter_value,
                            "Recovered": best_fit_parameter_value,
                        }
                    )

        # Create dataframe and return it
        likelihood = pd.DataFrame(likelihood_data)

        return likelihood

    def _random_parameters(
        parameter_fit_bounds, random_number_generator=numpy.random.default_rng()
    ):
        """Returns a dictionary of parameter-value pairs where the value is random within the specified fit bounds.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param random_number_generator: The random number generator which controls how the 'true' parameter values are
            generated, defaults to numpy.random.default_rng()
        :type random_number_generator: numpy.random.Generator, optional
        :return: A dictionary of parameter-value pairs where the value is random within the specified fit bounds
        :rtype: dict
        """
        # Data container
        random_parameters = {}

        # If parameters and their fit bounds were specified
        if len(parameter_fit_bounds) > 0:

            # For each parameter and their fit bounds
            for (
                current_parameter_name,
                current_parameter_fit_bounds,
            ) in parameter_fit_bounds.items():

                # Compute a random parameter value within the fit bounds
                random_parameters[
                    current_parameter_name
                ] = random_number_generator.uniform(*current_parameter_fit_bounds)

        # Return the parameter-value pairs
        return random_parameters

    def _simulate(self, user=None, random_number_generator=numpy.random.default_rng()):
        """Returns a DataFrame containing the behavioral data from simulating the given task with
        the specified user.

        :param user: The user to use for the simulation (if None is specified, will use the user
            of the bundle (i.e. self.user)), defaults to None
        :type user: core.agents.BaseAgent, optional
        :param random_number_generator: The random number generator which controls how the 'true' parameter values are
            generated, defaults to numpy.random.default_rng()
        :type random_number_generator: numpy.random.Generator, optional
        :return: A DataFrame containing the behavioral data from simulating the given task with
            the specified user
        :rtype: pandas.DataFrame
        """
        # Bundle definition
        user_to_use_for_simulation = user if user is not None else self.user
        bundle = Bundle(task=self.task, user=user_to_use_for_simulation)

        # Reset the bundle to default values
        bundle.reset()

        # Seed the policy of the agent
        bundle.user.policy.rng = random_number_generator

        # Data container
        data = []

        # Flag whether the task has been completed
        done = False

        # While the task is not finished...
        while not done:

            # Save the current round number
            round = bundle.task.round

            # Simulate a round of the user executing the task
            _, rewards, done = bundle.step()

            # Save the action that the artificial agent made
            action_values = copy(bundle.user.policy.action_state["action"].values[0])

            # Store this round's data
            data.append(
                {
                    "time": round,
                    "action": action_values,
                    "reward": rewards["first_task_reward"],
                }
            )

        # When the task is done, create and return DataFrame
        simulated_data = pd.DataFrame(data)
        return simulated_data

    def best_fit_parameters(
        task,
        user_class,
        parameter_fit_bounds,
        data,
        random_number_generator=numpy.random.default_rng(),
        **kwargs,
    ):
        """Returns a list of the parameters with their best-fit values based on the supplied data.

        :param task: The interaction task to be performed
        :type task: core.interactiontask.InteractionTask
        :param user_class: The user class to find best-fit parameters for
        :type user_class: core.agents.BaseAgent
        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param data: The behavioral data to infer the best-fit parameters from
        :type data: pandas.DataFrame
        :param random_number_generator: The random number generator which controls how the 'true' parameter values are
            generated, defaults to numpy.random.default_rng()
        :type random_number_generator: numpy.random.Generator, optional
        :return: A list of the parameters with their best-fit values based on the supplied data
        :rtype: list
        """
        # If no parameters are specified...
        if not len(parameter_fit_bounds) > 0:
            # Calculate the negative log likelihood for the data without parameters...
            best_parameters = []
            ll = ModelChecks._log_likelihood(
                task=task,
                user_class=user_class,
                parameter_values=best_parameters,
                data=data,
            )
            best_objective_value = -ll

            # ...and return an empty list and the negative log-likelihood
            return best_parameters, best_objective_value

        # Define an initital guess
        random_initial_guess = ModelChecks._random_parameters(
            parameter_fit_bounds=parameter_fit_bounds,
            random_number_generator=random_number_generator,
        )
        initial_guess = [
            parameter_value for _, parameter_value in random_initial_guess.items()
        ]

        # Run the optimizer
        res = scipy.optimize.minimize(
            fun=ModelChecks._objective,
            x0=initial_guess,
            bounds=[fit_bounds for _, fit_bounds in parameter_fit_bounds.items()],
            args=(task, user_class, data),
            **kwargs,
        )

        # Make sure that the optimizer ended up with success
        assert res.success

        # Get the best parameter value from the result
        best_parameter_values = res.x
        best_objective_value = res.fun

        # Construct a list for the best parameters and return it
        best_parameters = [
            (current_parameter_name, best_parameter_values[current_parameter_index])
            for current_parameter_index, (current_parameter_name, _) in enumerate(
                parameter_fit_bounds.items()
            )
        ]

        return best_parameters, best_objective_value

    def _log_likelihood(task, user_class, parameter_values, data):
        """Returns the log-likelihood of the specified parameter values given the provided data.

        :param task: The interaction task to be performed
        :type task: core.interactiontask.InteractionTask
        :param user_class: The user class to compute the log-likelihood for
        :type user_class: core.agents.BaseAgent
        :param parameter_values: A list of the parameter values to compute the log-likelihood for
        :type parameter_values: list
        :param data: The behavioral data to compute the log-likelihood for
        :type data: pandas.DataFrame
        :return: The log-likelihood of the specified parameter values given the provided data
        :rtype: float
        """
        # Data container
        ll = []

        # Create a new agent with the current parameters
        agent = None
        if not len(parameter_values) > 0:
            agent = user_class()
        else:
            agent = user_class(*parameter_values)

        # Bundle definition
        bundle = Bundle(task=task, user=agent)

        bundle.reset()

        # Simulate the task
        for _, row in data.iterrows():

            # Get action and success for t
            action_values, reward = row["action"], row["reward"]
            action = agent.policy.new_action
            action["values"] = action_values

            # Get probability of this action
            p = agent.policy.compute_likelihood(action, agent.observation)

            # Compute log
            log = numpy.log(p + numpy.finfo(float).eps)
            ll.append(log)

            # Make user take specified action
            _, rewards, _ = bundle.step(action)

            # Compare simulated and resulting reward
            failure_message = """The provided user action did not yield the same reward from the task.
            Maybe there is some randomness involved that could be solved by seeding."""
            assert reward == rewards["first_task_reward"], failure_message

        return numpy.sum(ll)

    def _objective(parameter_values, task, user_class, data):
        """Returns the negative log-likelihood of the specified parameter values given the provided data.

        :param parameter_values: A list of the parameter values to compute the log-likelihood for
        :type parameter_values: list
        :param task: The interaction task to be performed
        :type task: core.interactiontask.InteractionTask
        :param user_class: The user class to calculate the negative log-likelihood for
        :type user_class: core.agents.BaseAgent
        :param data: The behavioral data to compute the log-likelihood for
        :type data: pandas.DataFrame
        :return: The negative log-likelihood of the specified parameter values given the provided data
        :rtype: float
        """
        # Since we will look for the minimum,
        # let's return -LLS instead of LLS
        return -ModelChecks._log_likelihood(
            task=task,
            user_class=user_class,
            parameter_values=parameter_values,
            data=data,
        )

    def _correlations_plot(parameter_fit_bounds, data, statistics=None, kind="reg"):
        """Plot the correlation between the true and recovered parameters.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be
            used to generate the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param data: The correlation data including each parameter value used and recovered
        :type data: pandas.DataFrame
        :param kind: The kind of plot to generate (one of "reg" or "scatter"), defaults to "reg"
        :type kind: str
        :param statistics: The correlation statistics including whether the parameters were recoverable for some
            specified fit bounds, defaults to None
        :type statistics: pandas.DataFrame
        """
        # Containers
        param_names = []
        param_bounds = []

        # Store parameter names and fit bounds in separate lists
        for (parameter_name, fit_bounds) in parameter_fit_bounds.items():
            param_names.append(parameter_name)
            param_bounds.append(fit_bounds)

        # Calculate number of parameters
        n_param = len(parameter_fit_bounds)

        # Define colors
        colors = [f"C{i}" for i in range(n_param)]

        # Create fig and axes
        _, axes = plt.subplots(ncols=n_param, figsize=(10, 9))

        # For each parameter...
        for i in range(n_param):

            # Select ax
            ax = axes
            if n_param > 1:
                ax = axes[i]

            # Get param name
            p_name = param_names[i]

            # Set title
            ax.set_title(p_name)

            # Select only data related to the current parameter
            current_parameter_data = data[data["Parameter"] == p_name]

            # Depending on specified kind...
            if kind == "reg":
                # Create regression plot
                scatterplot = sns.regplot(
                    data=current_parameter_data,
                    x="Used to simulate",
                    y="Recovered",
                    scatter_kws=dict(alpha=0.5),
                    line_kws=dict(alpha=0.5),
                    color=colors[i],
                    ax=ax,
                )

            elif kind == "scatter":
                # Create scatter plot
                scatterplot = sns.scatterplot(
                    data=data[data["Parameter"] == p_name],
                    x="Used to simulate",
                    y="Recovered",
                    alpha=0.5,
                    color=colors[i],
                    ax=ax,
                )

            else:
                raise NotImplementedError("kind has to be one of 'reg' or 'scatter'")

            # Plot identity function
            ax.plot(
                param_bounds[i],
                param_bounds[i],
                linestyle="--",
                alpha=0.5,
                color="black",
                zorder=-10,
            )

            # If correlation statistics were supplied...
            if statistics is not None:

                # Select only statistics related to the current parameter
                current_parameter_statistics = statistics[
                    statistics["parameter"] == p_name
                ]

                # Highlight recoverable areas (high, significant correlation)
                # Identify recoverable areas
                recoverable_areas = current_parameter_statistics.loc[
                    current_parameter_statistics["recoverable"]
                ]

                # For each recoverable area...
                for _, row in recoverable_areas.iterrows():

                    # Add a green semi-transparent rectangle to the background
                    ax.axvspan(
                        *ast.literal_eval(row["fit_bounds"]),
                        facecolor="g",
                        alpha=0.2,
                        zorder=-11,
                    )

            # Set axes limits
            ax.set_xlim(*param_bounds[i])
            ax.set_ylim(*param_bounds[i])

            # Square aspect
            ax.set_aspect(1)

        return scatterplot

    def _pearsons_r(
        self,
        parameter_fit_bounds,
        data,
        correlation_threshold=0.7,
        significance_level=0.05,
    ):
        """Returns a DataFrame containing the correlation value (Pearson's r) and significance for each parameter.

        :param parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type parameter_fit_bounds: dict
        :param data: The correlation data including each parameter value used and recovered
        :type data: pandas.DataFrame
        :param correlation_threshold: The threshold for Pearson's r value (i.e. the correlation coefficient between the used
            and recovered parameters), defaults to 0.7
        :type correlation_threshold: float, optional
        :param significance_level: The threshold for the p-value to consider the correlation significant, defaults to 0.05
        :type significance_level: float, optional
        """

        def pearson_r_data(parameter_name):
            """Returns a dictionary containing the parameter name and its correlation value and significance.

            :param parameter_name: The name of the parameter
            :type parameter_name: str
            :return: A dictionary containing the parameter name and its correlation value and significance
            :rtype: dict
            """
            # Get the elements to compare
            x = data.loc[data["Parameter"] == parameter_name, "Used to simulate"]
            y = data.loc[data["Parameter"] == parameter_name, "Recovered"]

            # Compute a Pearson correlation
            r, p = scipy.stats.pearsonr(x, y)

            # Return
            pearson_r_data = {
                "parameter": parameter_name,
                "r": r,
                f"r>{correlation_threshold}": r > correlation_threshold,
                "p": p,
                f"p<{significance_level}": p < significance_level,
                "recoverable": (r > correlation_threshold) and (p < significance_level),
            }

            return pearson_r_data

        # Compute correlation data
        correlation_data = [
            pearson_r_data(parameter_name)
            for parameter_name, fit_bounds in parameter_fit_bounds.items()
            if fit_bounds[1] - fit_bounds[0] > 1e-12
        ]

        # Create dataframe
        pearsons_r = pd.DataFrame(correlation_data)

        return pearsons_r

    @dataclass
    class ModelRecoveryTestResult:
        """Represents the results of a test for model recovery."""

        confusion_data: pd.DataFrame
        """The 'true' (i.e. actually simulated) and recovered models"""

        robustness_statistics: pd.DataFrame
        """Robustness statistics (i.e. precision, recall, F1-score) for the recovery of each model"""

        success: bool
        """`True` if the F1-score for all models exceeded the supplied threshold, `False` otherwise"""

        plot: matplotlib.axes.Axes
        """The heatmap displaying the 'true' (i.e. actually simulated) and recovered models (i.e. the confusion matrix)"""

        f1_threshold: float
        """The threshold for F1-score to consider the recovery successful for a model"""

        n_simulations_per_model: int
        """The number of agents that were simulated (i.e. the population size) for each model"""

        method: str
        """The metric by which the recovered model was chosen"""

    def test_model_recovery(
        self,
        other_competing_models,
        this_parameter_fit_bounds,
        f1_threshold=0.7,
        n_simulations_per_model=20,
        method="BIC",
        seed=None,
        **kwargs,
    ):
        """Returns whether the bundle's user model can be recovered from simulated data using the specified competing models
        meeting the specified F1-score threshold (only available for users with a policy that has a compute_likelihood method).

        It simulates n_simulations agents for each of the user's class and the competing models using random parameters within the supplied
        parameter_fit_bounds, executes the provided task and tries to recover the user's best-fit parameters from the simulated data. Each of
        the best-fit models is then evaluated for fit using the BIC-score. The model recovery is then evaluated using recall, precision and
        the F1-score which is finally evaluated against the specified threshold.

        :param other_competing_models: A list of dictionaries for the other competing models including their parameter fit bounds (i.e. their names,
            their minimum and maximum values) that will be used for simulation (example: `[{"model": UserClass, "parameter_fit_bounds": {"alpha": (0., 1.), ...}}, ...]`)
        :type other_competing_models: list(dict)
        :param this_parameter_fit_bounds: A dictionary of the parameter names, their minimum and maximum values that will be used to generate
            the random parameter values for simulation of the bundle user class (example: `{"alpha": (0., 1.), "beta": (0., 20.)}`)
        :type this_parameter_fit_bounds: dict
        :param f1_threshold: The threshold for F1-score, defaults to 0.7
        :type f1_threshold: float, optional
        :param n_simulations_per_model: The number of agents to simulate (i.e. the population size) for each model, defaults to 20
        :type n_simulations_per_model: int, optional
        :param method: The metric by which to choose the recovered model, should be one of "BIC" (Bayesian Information Criterion)
            or "AIC" (Akaike Information Criterion), defaults to "BIC"
        :type method: str, optional
        :param seed: The seed for the random number generator which controls how the 'true' parameter values are generated, defaults to None
        :type seed: int, optional
        :return: `True` if the F1-score for the model recovery meets the supplied threshold, `False` otherwise
        :rtype: bool
        """
        # Transform this_parameter_fit_bounds to empty dict if falsy (e.g. [], {}, None, False)
        if not this_parameter_fit_bounds:
            this_parameter_fit_bounds = {}

        # All user models that are competing
        all_user_classes = other_competing_models + [
            {
                "model": self.user.__class__,
                "parameter_fit_bounds": this_parameter_fit_bounds,
            }
        ]

        # Calculate the confusion matrix between used and recovered models for and from simulation
        confusion_matrix = self._confusion_matrix(
            all_user_classes=all_user_classes,
            n_simulations=n_simulations_per_model,
            method=method,
            seed=seed,
            **kwargs,
        )

        # Create the confusion matrix
        confusion_matrix_plot = ModelChecks._confusion_matrix_plot(
            data=confusion_matrix
        )

        # Get the model names
        model_names = [m["model"].__name__ for m in all_user_classes]

        # Compute the model recovery statistics (recall, precision, f1)
        robustness = self._robustness_statistics(
            model_names=model_names, f1_threshold=f1_threshold, data=confusion_matrix
        )

        # Check that all correlations meet the threshold and are significant and return the result
        all_f1_meet_threshold = robustness[f"f1>{f1_threshold}"].all()

        # Create the result and return it
        result = ModelChecks.ModelRecoveryTestResult(
            confusion_data=confusion_matrix,
            robustness_statistics=robustness,
            success=all_f1_meet_threshold,
            plot=confusion_matrix_plot,
            f1_threshold=f1_threshold,
            n_simulations_per_model=n_simulations_per_model,
            method=method,
        )
        return result

    def _confusion_matrix(
        self, all_user_classes, n_simulations=20, method="BIC", seed=None, **kwargs
    ):
        """Returns a DataFrame with the model recovery data (used to simulate vs recovered model) based on the BIC-score.

        :param all_user_classes: The user models that are competing and can be recovered (example: `[{"model": UserClass, "parameter_fit_bounds": {"alpha": (0., 1.), ...}}, ...]`)
        :type all_user_classes: list(dict)
        :param n_simulations: The number of agents to simulate (i.e. the population size) for each model, defaults to 20
        :type n_simulations: int, optional
        :param method: The metric by which to choose the recovered model, should be one of "BIC" (Bayesian Information Criterion)
            or "AIC" (Akaike Information Criterion), defaults to "BIC"
        :type method: str, optional
        :param seed: The seed for the random number generator which controls how the 'true' parameter values are generated, defaults to None
        :type seed: int, optional
        :return: A DataFrame with the model recovery data (used to simulate vs recovered model) based on the BIC-score
        :rtype: pandas.DataFrame
        """
        # Number of models
        n_models = len(all_user_classes)

        # Data container
        confusion_matrix = numpy.zeros((n_models, n_models))

        # Random number generator
        random_number_generator = numpy.random.default_rng(seed)

        # Set up progress bar
        with tqdm(total=n_simulations * n_models, file=sys.stdout) as pbar:

            # Loop over each model
            for i, user_class_to_sim in enumerate(all_user_classes):

                m_to_sim = user_class_to_sim["model"]
                parameters_for_sim = user_class_to_sim["parameter_fit_bounds"]

                for _ in range(n_simulations):

                    # Generate a random agent
                    attributes_from_self_user = {
                        attr: getattr(self.user, attr)
                        for attr in dir(self.user)
                        if attr in inspect.signature(self.user.__class__).parameters
                    }
                    random_agent = None
                    if not len(parameters_for_sim) > 0:
                        random_agent = (
                            m_to_sim(**attributes_from_self_user)
                            if m_to_sim == self.user.__class__
                            else m_to_sim()
                        )
                    else:
                        random_parameters = ModelChecks._random_parameters(
                            parameter_fit_bounds=parameters_for_sim,
                            random_number_generator=random_number_generator,
                        )
                        random_agent = (
                            m_to_sim(
                                **{**attributes_from_self_user, **random_parameters}
                            )
                            if m_to_sim == self.user.__class__
                            else m_to_sim(**random_parameters)
                        )

                    # Simulate the task
                    simulated_data = self._simulate(
                        user=random_agent,
                        random_number_generator=random_number_generator,
                    )

                    # Determine best-fit models
                    best_fit_models, _ = ModelChecks.best_fit_models(
                        task=self.task,
                        all_user_classes=all_user_classes,
                        data=simulated_data,
                        method=method,
                        random_number_generator=random_number_generator,
                        **kwargs,
                    )

                    # Get index(es) of models that get best score (e.g. BIC)
                    idx_min = [
                        user_class_index
                        for user_class_index, user_class in enumerate(all_user_classes)
                        if user_class["model"].__name__
                        in [
                            best_fit_model["model"].__name__
                            for best_fit_model in best_fit_models
                        ]
                    ]

                    # Add result in matrix
                    confusion_matrix[i, idx_min] += 1 / len(idx_min)

                    # Update progress bar
                    pbar.update(1)

        # Get the model names
        model_names = [m["model"].__name__ for m in all_user_classes]

        # Create dataframe
        confusion = pd.DataFrame(
            confusion_matrix, index=model_names, columns=model_names
        )

        return confusion

    def best_fit_models(
        task,
        all_user_classes,
        data,
        method="BIC",
        random_number_generator=numpy.random.default_rng(),
        **kwargs,
    ):
        """Returns a list of the recovered best-fit model(s) based on the BIC-score and
        a list of dictionaries containing the BIC-score for all competing models.

        :param task: The interaction task to be performed
        :type task: core.interactiontask.InteractionTask
        :param all_user_classes: The user models that are competing and can be
        recovered (example: `[{"model": UserClass, "parameter_fit_bounds": {"alpha": (0., 1.), ...}}, ...]`)
        :type all_user_classes: list(dict)
        :param data: The behavioral data as a DataFrame with the columns "time", "action" and "reward"
        :type data: pandas.DataFrame
        :param method: The metric by which to choose the recovered model, should be one of "BIC" (Bayesian Information Criterion)
            or "AIC" (Akaike Information Criterion), defaults to "BIC"
        :type method: str, optional
        :param random_number_generator: The random number generator which controls how the initial guess for the parameter values are generated, defaults to numpy.random.default_rng()
        :type random_number_generator: numpy.Generator, optional
        :return: A list of the recovered best-fit model(s) based on the BIC-score and
        a list of dictionaries containing the BIC-score for all competing models
        :rtype: tuple[list[str], list[dict]]
        """
        data_has_necessary_columns = set(["time", "action", "reward"]).issubset(
            data.columns
        )
        if not data_has_necessary_columns:
            raise ValueError(
                "data argument must have the columns 'time', 'action' and 'reward'."
            )

        # Number of models
        n_models = len(all_user_classes)

        # Container for BIC scores
        bs_scores = numpy.zeros(n_models)

        # Container for best-fit parameters
        all_best_fit_parameters = []

        # For each model
        for k, user_class_to_fit in enumerate(all_user_classes):

            m_to_fit = user_class_to_fit["model"]
            parameters_for_fit = user_class_to_fit["parameter_fit_bounds"]

            # Determine best-fit parameter values
            (
                best_fit_parameters,
                best_fit_objective_value,
            ) = ModelChecks.best_fit_parameters(
                task=task,
                user_class=m_to_fit,
                parameter_fit_bounds=parameters_for_fit,
                data=data,
                random_number_generator=random_number_generator,
                **kwargs,
            )

            # Get log-likelihood for best param
            ll = -best_fit_objective_value

            # Compute the comparison metric score (e.g. BIC)
            n_param_m_to_fit = len(parameters_for_fit)

            if method == "BIC":
                bs_scores[k] = bic(log_likelihood=ll, k=n_param_m_to_fit, n=len(data))
            elif method == "AIC":
                bs_scores[k] = aic(log_likelihood=ll, k=n_param_m_to_fit)
            else:
                raise NotImplementedError("method has to be one of 'BIC' or 'AIC'")

            # Store best-fit parameters
            all_best_fit_parameters.append(best_fit_parameters)

        # Get minimum value for BIC/AIC (min => best)
        min_score = numpy.min(bs_scores)

        # Get index(es) of models that get best BIC/AIC
        idx_min = numpy.flatnonzero(bs_scores == min_score)

        # Identify best-fit models
        best_fit_models = []
        for i in idx_min:
            best_fit_model = copy(all_user_classes[i])
            best_fit_model["parameters"] = all_best_fit_parameters[i]
            best_fit_models.append(best_fit_model)

        # Create list for all models and their BIC/AIC scores
        all_bic_scores = [
            {user_class["model"].__name__: bs_scores[i]}
            for i, user_class in enumerate(all_user_classes)
        ]

        # Return best-fit models and all BIC/AIC scores
        return best_fit_models, all_bic_scores

    def _confusion_matrix_plot(data):
        """Returns a plot of the confusion matrix for the model recovery comparison.

        :param data: The confusion matrix (model used to simulate vs recovered model) as a DataFrame
        :type data: pandas.DataFrame
        :return: A plot of the confusion matrix for the model recovery comparison
        :rtype: matplotlib.axes.Axes
        """
        # Create figure and axes
        _, ax = plt.subplots(figsize=(12, 10))

        # Display the results using a heatmap
        heatmap = sns.heatmap(data=data, cmap="viridis", annot=True, ax=ax)

        # Set x-axis and y-axis labels
        ax.set_xlabel("Recovered")
        ax.set_ylabel("Used to simulate")

        return heatmap

    def _recall(model_name, data):
        """Returns the recall value and its confidence interval for the given model and confusion matrix.

        :param model_name: The name of the model to compute the recall for
        :type model_name: str
        :param data: The confusion matrix as a DataFrame
        :type data: pandas.DataFrame
        :return: The recall value and its confidence interval for the given model and confusion matrix
        :rtype: tuple[float, tuple[float, float]]
        """
        # Get the number of true positive
        k = data.at[model_name, model_name]

        # Get the number of true positive + false NEGATIVE
        n = numpy.sum(data.loc[model_name])

        # Compute the recall and return it
        recall = k / n

        # Compute the confidence interval
        ci_recall = statsmodels.stats.proportion.proportion_confint(count=k, nobs=n)

        return recall, ci_recall

    def _precision(model_name, data):
        """Returns the precision value and its confidence interval for the given model and confusion matrix.

        :param model_name: The name of the model to compute the precision for
        :type model_name: str
        :param data: The confusion matrix as a DataFrame
        :type data: pandas.DataFrame
        :return: The precision value and its confidence interval for the given model and confusion matrix
        :rtype: tuple[float, tuple[float, float]]
        """
        # Get the number of true positive
        k = data.at[model_name, model_name]

        # Get the number of true positive + false POSITIVE
        n = numpy.sum(data[model_name])

        # Compute the precision
        precision = k / n

        # Compute the confidence intervals
        ci_pres = statsmodels.stats.proportion.proportion_confint(k, n)

        return precision, ci_pres

    def _robustness_statistics(self, model_names, f1_threshold, data):
        """Returns a DataFrame with the robustness statistics (precision, recall, F1-score) based on the
        supplied confusion data and user models.

        :param model_names: The names of the user models that are competing and could be recovered
        :type all_user_classes: list(str)
        :param f1_threshold: The threshold for F1-score, defaults to 0.7
        :type f1_threshold: float, optional
        :param data: The confusion matrix as a DataFrame
        :type data: pandas.DataFrame
        :return: A DataFrame with the robustness statistics (precision, recall, F1-score) based on the
            supplied confusion data and user models
        :rtype: pandas.DataFrame
        """
        # Results container
        row_list = []

        # For each model...
        for m in model_names:

            # Compute the recall
            recall, ci_recall = ModelChecks._recall(model_name=m, data=data)

            # Compute the precision and confidence intervals
            precision, ci_pres = ModelChecks._precision(model_name=m, data=data)

            # Compute the f score
            f_score = f1(precision, recall)

            # Backup
            row_list.append(
                {
                    "model": m,
                    "Recall": recall,
                    "Recall [CI]": ci_recall,
                    "Precision": precision,
                    "Precision [CI]": ci_pres,
                    "F1 score": f_score,
                    f"f1>{f1_threshold}": f_score > f1_threshold,
                }
            )

        # Create dataframe and display it
        stats = pd.DataFrame(row_list, index=model_names)

        return stats

    @dataclass
    class RecoverableParameterRangesTestResult:
        """Represents the results of a test for recoverable parameter ranges."""

        correlation_data: pd.DataFrame
        """The 'true' and recovered parameter value pairs"""

        correlation_statistics: pd.DataFrame
        """The correlation statistics (i.e. the correlation coefficient and its p-value) for the parameter recovery of each sub-range"""

        plot: matplotlib.axes.Axes
        """The scatterplot displaying the 'true' and recovered parameter values for each parameter, highlighting the recoverable ranges"""

        correlation_threshold: float
        """The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered parameters)"""

        significance_level: float
        """The threshold for the p-value to consider the correlation significant"""

        n_simulations_per_sub_range: int
        """The number of agents that were simulated (i.e. the population size) for each tested sub-range to identify the
        recoverable parameter ranges"""

        recoverable_parameter_ranges: OrderedDict
        """The ranges where parameter recovery meets the required thresholds for all parameters (example: [OrderedDict([('alpha', (0.0, 0.3)), ('beta', (0.0, 0.2))], …])"""

        recovered_parameter_correlation_threshold: float = None
        """The threshold for Pearson's r value (i.e. the correlation coefficient between the recovered parameters) to consider them correlated"""

        @property
        def success(self):
            """`True` if recoverable parameter ranges could be identified, `False` otherwise"""
            return len(self.recoverable_parameter_ranges) > 0

    def test_recoverable_parameter_ranges(
        self,
        parameter_ranges,
        correlation_threshold=0.7,
        significance_level=0.05,
        n_simulations_per_sub_range=100,
        recovered_parameter_correlation_threshold=0.7,
        seed=None,
    ):
        """Returns the ranges for each specified parameter of the bundle's user model where parameter recovery meets the required thresholds
        for all parameters.

        :param parameter_ranges: A dictionary of the parameter names and their respective ranges that will be used to generate
            the random parameter values for simulation of the bundle user class (example: `{"alpha": numpy.linspace(0., 1., num=10), "beta": range(0., 20., num=5)}`)
        :type parameter_ranges: dict[str, ndarray]
        :param correlation_threshold: The threshold for Pearson's r value (i.e. the correlation coefficient between the used and recovered
            parameters) for each sub-range, defaults to 0.7
        :type correlation_threshold: float, optional
        :param significance_level: The threshold for the p-value to consider the correlation significant, defaults to 0.05
        :type significance_level: float, optional
        :param n_simulations_per_sub_range: The number of agents to simulate (i.e. the population size) for each sub-range, defaults to 100
        :type n_simulations_per_sub_range: int, optional
        :param recovered_parameter_correlation_threshold: The threshold for Pearson's r value between the recovered parameters to consider them correlated, defaults to 0.7
        :type recovered_parameter_correlation_threshold: float, optional
        :param seed: The seed for the random number generator which controls how the 'true' parameter values are generated, defaults to None
        :type seed: int, optional
        :return: The ranges where parameter recovery meets the required thresholds for all parameters (example: [OrderedDict([('alpha', (0.0, 0.3)), ('beta', (0.0, 0.2))], …])
        :rtype: collections.OrderedDict
        """
        # Random number generator
        rng = numpy.random.default_rng(seed)

        # Transform the specified dict of parameter fit bounds into an OrderedDict
        # based on the order of parameters in the user class constructor
        ordered_parameter_ranges = order_class_parameters_by_signature(
            self.user.__class__, parameter_ranges
        )

        # Create containers for correlation data, statistics and recoverable ranges
        all_correlation_data = []
        all_correlation_statistics = []
        recoverable_parameter_ranges = OrderedDict(
            [(parameter_name, []) for parameter_name in parameter_ranges.keys()]
        )

        # Create fit bounds (i.e. min/max value pairs) from the specified ranges
        all_fit_bounds = ModelChecks._all_fit_bounds_from_parameter_ranges(
            ordered_parameter_ranges
        )

        # Determine maximum fit bounds for parameters from the specified ranges
        max_fit_bounds = ModelChecks._maximum_parameter_fit_bounds_from_ranges(
            ordered_parameter_ranges
        )

        # Set up progress bar
        n_fit_bounds = sum([len(fit_bounds) for fit_bounds in all_fit_bounds.values()])
        with tqdm(
            total=n_fit_bounds * n_simulations_per_sub_range, file=sys.stdout
        ) as pbar:

            # For each parameter...
            for (
                parameter_name,
                all_fit_bounds_for_current_parameter,
            ) in all_fit_bounds.items():

                # For each fit bound...
                for fit_bounds in all_fit_bounds_for_current_parameter:

                    # Container for correlation data
                    correlation_data = []

                    # For the specified number of simulations per sub-range...
                    for subject_index in range(n_simulations_per_sub_range):

                        # Generate a random value for the current parameter within fit bounds,
                        # generate random values for the other parameters in entire range
                        parameter_fit_bounds = {
                            current_parameter_name: fit_bounds
                            if current_parameter_name == parameter_name
                            else maximum_fit_bounds_tuple
                            for current_parameter_name, maximum_fit_bounds_tuple in max_fit_bounds.items()
                        }
                        # Generate a random agent
                        attributes_from_self_user = {
                            attr: getattr(self.user, attr)
                            for attr in dir(self.user)
                            if attr in inspect.signature(self.user.__class__).parameters
                        }
                        random_agent = None
                        if not len(parameter_fit_bounds) > 0:
                            random_agent = self.user.__class__(
                                **attributes_from_self_user
                            )
                        else:
                            random_parameters = ModelChecks._random_parameters(
                                parameter_fit_bounds=parameter_fit_bounds,
                                random_number_generator=rng,
                            )
                            random_agent = self.user.__class__(
                                **{**attributes_from_self_user, **random_parameters}
                            )

                        # Simulate behavior and store parameter values
                        simulated_data = self._simulate(
                            user=random_agent, random_number_generator=rng
                        )

                        # Perform test for parameter recovery only for current parameter (other parameter values treated as known)
                        parameter_fit_bounds = {
                            current_parameter_name: (
                                random_parameters[current_parameter_name] - 1e-13 / 2,
                                random_parameters[current_parameter_name] + 1e-13 / 2,
                            )
                            if current_parameter_name != parameter_name
                            else current_parameter_fit_bounds
                            for current_parameter_name, current_parameter_fit_bounds in parameter_fit_bounds.items()
                        }
                        # Determine best-fit parameter values
                        best_fit_parameters, _ = ModelChecks.best_fit_parameters(
                            task=self.task,
                            user_class=self.user.__class__,
                            parameter_fit_bounds=parameter_fit_bounds,
                            data=simulated_data,
                            random_number_generator=rng,
                        )
                        best_fit_parameters_dict = OrderedDict(best_fit_parameters)

                        true_parameter_value = random_parameters[parameter_name]
                        recovered_parameter_value = best_fit_parameters_dict[
                            parameter_name
                        ]

                        correlation_data.append(
                            {
                                "Subject": subject_index + 1,
                                "Parameter": parameter_name,
                                "Used to simulate": true_parameter_value,
                                "Recovered": recovered_parameter_value,
                                "Fit bounds": str(fit_bounds),
                            }
                        )

                        pbar.update(1)

                    # Transform data into DataFrame
                    correlation_data = pd.DataFrame(correlation_data)

                    # Compute the correlation metric Pearson's r and its significance for each parameter pair and return it
                    correlation_statistics = self._pearsons_r(
                        parameter_fit_bounds=parameter_fit_bounds,
                        data=correlation_data,
                        correlation_threshold=correlation_threshold,
                        significance_level=significance_level,
                    )

                    # Add fit bound information to correlation statistics
                    correlation_statistics["fit_bounds"] = str(fit_bounds)

                    # Check that the correlation meets the threshold and is significant
                    parameter_can_be_recovered = (
                        ModelChecks._correlations_meet_thresholds(
                            correlation_statistics=correlation_statistics,
                            correlation_threshold=correlation_threshold,
                            significance_level=significance_level,
                        )
                    )

                    # Store the data and statistics
                    all_correlation_data.append(correlation_data)
                    all_correlation_statistics.append(correlation_statistics)

                    # If the test was successful...
                    if parameter_can_be_recovered:

                        # Store the fit bounds
                        recoverable_parameter_ranges[parameter_name].append(fit_bounds)

        # Concat data and statistics
        all_correlation_data = pd.concat(all_correlation_data)
        all_correlation_statistics = pd.concat(all_correlation_statistics)

        # Create scatterplot of the recoverable parameter fit bounds test
        scatterplot = ModelChecks._recoverable_fit_bounds_result_plot(
            ordered_parameter_ranges=ordered_parameter_ranges,
            correlation_data=all_correlation_data,
            correlation_statistics=all_correlation_statistics,
        )

        # Create result and return it
        result = ModelChecks.RecoverableParameterRangesTestResult(
            correlation_data=all_correlation_data,
            correlation_statistics=all_correlation_statistics,
            plot=scatterplot,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
            n_simulations_per_sub_range=n_simulations_per_sub_range,
            recoverable_parameter_ranges=recoverable_parameter_ranges,
            recovered_parameter_correlation_threshold=recovered_parameter_correlation_threshold
            if len(parameter_ranges) > 1
            else None,
        )
        return result

    def _all_fit_bounds_from_parameter_ranges(ordered_parameter_ranges):
        """Returns an ordered dictionary containing lists with the lower-/upper-bound combinations from the specified
        parameter ranges (e.g. {"alpha": [(0.0, 0.2), (0.2, 0.4), ...], "beta": [(0.5, 1.0), (1.0, 1.5), ...], ...}).

        :param parameter_ranges: A dictionary of the parameter names and their respective ranges that will be used to generate
            the random parameter values for simulation of the bundle user class (example: `{"alpha": numpy.linspace(0., 1., num=10), "beta": range(0., 20., num=5)}`)
        :type parameter_ranges: dict[str, ndarray]
        :param ordered_parameter_ranges: An OrderedDict of the parameter names and their respective ranges that will be used to generate
            the random parameter values for simulation of the bundle user class
        :type ordered_parameter_ranges: collections.OrderedDict
        :return: An ordered dictionary containing lists with the lower-/upper-bound combinations from the specified
            parameter ranges (e.g. {"alpha": [(0.0, 0.2), (0.2, 0.4), ...], "beta": [(0.5, 1.0), (1.0, 1.5), ...], ...})
        :rtype: OrderedDict[tuple[str, list[tuple[float, float]]]]
        """
        # Container for the formatted parameter ranges (i.e. name and fit bounds for each combination)
        formatted_parameter_ranges = OrderedDict()

        # For each parameter...
        for parameter_name, parameter_range in ordered_parameter_ranges.items():
            # Container for the fit bounds for this step
            single_parameter_ranges = []
            # For each step in the range...
            for i in range(0, len(parameter_range) - 1):

                # Determine fit bounds
                lower_bound = parameter_range[i]
                upper_bound = parameter_range[i + 1]
                fit_bounds = (lower_bound, upper_bound)

                # Store results for this step
                single_parameter_ranges.append(fit_bounds)

            # Store results for this parameter
            formatted_parameter_ranges[parameter_name] = single_parameter_ranges

        return formatted_parameter_ranges

    def _maximum_parameter_fit_bounds_from_ranges(parameter_ranges):
        """Returns a dictionary containing each parameter name and its total range (i.e. minimum and maximum value) from the specified ranges.

        :param parameter_ranges: A dictionary of the parameter names and their respective ranges that will be used to generate
            the random parameter values for simulation of the bundle user class (example: `{"alpha": numpy.linspace(0., 1., num=10), "beta": range(0., 20., num=5)}`)
        :type parameter_ranges: dict[str, ndarray]
        :return: A dictionary containing each parameter name and its total range (i.e. minimum and maximum value) from the specified ranges
        :rtype: dict[str, tuple[float, float]]
        """
        return {
            parameter_name: (parameter_range.min(), parameter_range.max())
            for parameter_name, parameter_range in parameter_ranges.items()
        }

    def _recoverable_fit_bounds_result_plot(
        ordered_parameter_ranges, correlation_data, correlation_statistics
    ):
        """Returns a plot for the correlations of the 'true' and recovered parameter values and prints the correlation statistics.

        :param ordered_parameter_ranges: The parameter ranges (incl. steps) to use as basis for the parameter fit bounds
        :type ordered_parameter_ranges: collections.OrderedDict
        :param correlation_data: The 'true' and recovered parameter value pairs
        :type correlation_data: pandas.DataFrame
        :param correlation_statistics: The correlation statistics for the parameter recovery
        :type correlation_statistics: pandas.DataFrame
        :return: A plot for the correlations of the 'true' and recovered parameter values and prints the correlation statistics
        :rtype: matplotlib.axes.Axes
        """
        # Format the specified parameter ranges into parameter fit bounds (i.e. without step size)
        parameter_fit_bounds = ModelChecks._formatted_parameter_fit_bounds(
            ordered_parameter_ranges
        )

        # Plot the correlations of the 'true' and recovered parameter values
        scatterplot = ModelChecks._correlations_plot(
            parameter_fit_bounds=parameter_fit_bounds,
            data=correlation_data,
            statistics=correlation_statistics,
            kind="scatter",
        )

        # Print the correlation statistics for the 'true' and recovered parameter values per sub-range
        ModelChecks._print_correlation_statistics(correlation_statistics)

        return scatterplot

    def _formatted_parameter_fit_bounds(ordered_parameter_ranges):
        """Returns an OrderedDict of each parameter and its associated fit bounds (i.e. minimum and maximum value) from
        an OrderedDict specifying the parameter ranges.

        :param ordered_parameter_ranges: The parameter ranges (incl. steps) to use as basis for the parameter fit bounds
        :type ordered_parameter_ranges: collections.OrderedDict
        :return: An OrderedDict of each parameter and its associated fit bounds (i.e. minimum and maximum value)
        :rtype: collections.OrderedDict
        """
        parameter_fit_bounds = OrderedDict()
        for parameter_name, parameter_range in ordered_parameter_ranges.items():
            parameter_fit_bounds[parameter_name] = (
                parameter_range.min(),
                parameter_range.max(),
            )
        return parameter_fit_bounds

    def _print_correlation_statistics(correlation_statistics):
        """Prints a table with the correlation statistics to the standard output.

        :param correlation_statistics: The correlation statistics to print
        :type correlation_statistics: pandas.DataFrame
        """
        correlation_statistics_table = tabulate(
            correlation_statistics, headers="keys", tablefmt="psql", showindex=False
        )
        print(correlation_statistics_table)
