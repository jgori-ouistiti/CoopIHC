from core.agents import DummyAssistant
from core.bundle import SinglePlayOperator, SinglePlayOperatorAuto

from tqdm import tqdm
import sys
from copy import copy
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.proportion


def _random_parameters(parameter_fit_bounds):
    random_parameters = {}
    if len(parameter_fit_bounds) > 0:
        for (current_parameter_name, current_parameter_fit_bounds) in parameter_fit_bounds.items():
            random_parameters[current_parameter_name] = np.random.uniform(
                *current_parameter_fit_bounds)
    return random_parameters


def _likelihood(task, operator_class, parameter_fit_bounds, population_size):
    # Data container
    likelihood_data = []

    # For each agent...
    for _ in tqdm(range(population_size), file=sys.stdout):

        # Generate a random agent
        random_agent = None
        if not len(parameter_fit_bounds) > 0:
            random_agent = operator_class()
        else:
            random_parameters = _random_parameters(
                parameter_fit_bounds=parameter_fit_bounds)
            random_agent = operator_class(**random_parameters)

        # Simulate the task
        simulated_data = _simulate(task=task, operator=random_agent)

        # Determine best-fit parameter values
        best_fit_parameters, _ = _best_fit_parameters(
            operator_class=operator_class, parameter_fit_bounds=parameter_fit_bounds, data=simulated_data, task=task)

        # Backup parameter values
        for parameter_index, (parameter_name, parameter_value) in enumerate(random_parameters.items()):
            _, best_fit_parameter_value = best_fit_parameters[parameter_index]
            likelihood_data.append({
                "Parameter": parameter_name,
                "Used to simulate": parameter_value,
                "Recovered": best_fit_parameter_value
            })

    # Create dataframe and return it
    likelihood = pd.DataFrame(likelihood_data)

    return likelihood


def _log_likelihood(parameter_values, operator_class, data, task):
    # Data container
    ll = []

    # Create a new agent with the current parameters
    agent = operator_class() if not len(
        parameter_values) > 0 else operator_class(*parameter_values)

    # Bundle definition
    bundle = SinglePlayOperator(task, agent)

    bundle.reset()

    # Simulate the task
    for _, row in data.iterrows():

        # Get choice and success for t
        choice, reward = row["choice"], row["reward"]

        # Get probability of this choice
        observation = agent.inference_engine.buffer[-1]
        p = agent.policy.compute_likelihood(choice, observation)

        # Compute log
        log = np.log(p + np.finfo(float).eps)
        ll.append(log)

        # Make operator take specified action
        _, resulting_reward, _, _ = bundle.step(operator_action=choice)

        # Compare simulated and resulting reward
        assert reward == resulting_reward, "The provided operator action did not yield the same reward from the task. Maybe there is some randomness involved that could be solved by seeding."

    return np.sum(ll)


def _objective(parameter_values, operator_class, data, task):
    # Since we will look for the minimum,
    # let's return -LLS instead of LLS
    negative_lls = - _log_likelihood(parameter_values=parameter_values,
                                     operator_class=operator_class, data=data, task=task)
    return negative_lls


def _best_fit_parameters(operator_class, parameter_fit_bounds, data, task):
    if not len(parameter_fit_bounds) > 0:
        best_parameters = []
        best_objective_value = - _log_likelihood(
            parameter_values=best_parameters, operator_class=operator_class, data=data, task=task)
        return best_parameters, best_objective_value

    # Define an initital guess
    random_initial_guess = _random_parameters(
        parameter_fit_bounds=parameter_fit_bounds)
    initial_guess = [parameter_value for _,
                     parameter_value in random_initial_guess.items()]

    # Run the optimizer
    res = scipy.optimize.minimize(
        fun=_objective,
        x0=initial_guess,
        bounds=[fit_bounds for _,
                fit_bounds in parameter_fit_bounds.items()],
        args=(operator_class, data, task))

    # Make sure that the optimizer ended up with success
    assert res.success

    # Get the best parameter value from the result
    best_parameter_values = res.x
    best_objective_value = res.fun

    best_parameters = [(current_parameter_name, best_parameter_values[current_parameter_index])
                       for current_parameter_index, (current_parameter_name, _) in enumerate(parameter_fit_bounds.items())]

    return best_parameters, best_objective_value


def _simulate(task, operator):
    # Bundle definition
    bundle = SinglePlayOperatorAuto(task, operator)

    bundle.reset()

    rewards = []
    choices = []

    done = False

    while not done:
        game_state, reward, done, _ = bundle.step()

        choice = copy(game_state["task_state"]["last_action"])

        choices.append(choice)
        rewards.append(reward)

    # If done or maximum number of steps reached
    # Create and return DataFrame
    simulated_data = pd.DataFrame({
        "time": np.arange(task.round + 1),
        "choice": choices,
        "reward": rewards
    })

    # Return the simulated data
    return simulated_data


def _pearsons_r(operator_class, parameter_fit_bounds, data, significance_level=0.05):
    def pearson_r_data(parameter_name):
        # Get the elements to compare
        x = data.loc[data["Parameter"] == parameter_name, "Used to simulate"]
        y = data.loc[data["Parameter"] == parameter_name, "Recovered"]

        # Compute a Pearson correlation
        r, p = scipy.stats.pearsonr(x, y)

        # Return
        pearson_r_data = {
            "parameter": parameter_name,
            "r": r,
            "p": p,
            f"p<{significance_level}": p < significance_level
        }

        return pearson_r_data

    # Compute correlation data
    correlation_data = [pearson_r_data(parameter_name)
                        for parameter_name, _ in parameter_fit_bounds.items()]

    # Create dataframe
    df_pearsons_r = pd.DataFrame(correlation_data)

    return df_pearsons_r


def _plot_correlations(parameter_fit_bounds, data):
    """Plot the correlation between the true and recovered parameters."""

    # Plot
    param_names = []
    param_bounds = []

    for (parameter_name, fit_bounds) in parameter_fit_bounds.items():
        param_names.append(parameter_name)
        param_bounds.append(fit_bounds)

    n_param = len(parameter_fit_bounds)

    # Define colors
    colors = [f'C{i}' for i in range(n_param)]

    # Create fig and axes
    _, axes = plt.subplots(ncols=n_param,
                           figsize=(10, 9))

    for i in range(n_param):

        # Select ax
        ax = axes
        if n_param > 1:
            ax = axes[i]

        # Get param name
        p_name = param_names[i]

        # Set title
        ax.set_title(p_name)

        # Create scatter
        sns.scatterplot(data=data[data["Parameter"] == p_name],
                        x="Used to simulate", y="Recovered",
                        alpha=0.5, color=colors[i],
                        ax=ax)

        # Plot identity function
        ax.plot(param_bounds[i], param_bounds[i],
                linestyle="--", alpha=0.5, color="black", zorder=-10)

        # Set axes limits
        ax.set_xlim(*param_bounds[i])
        ax.set_ylim(*param_bounds[i])

        # Square aspect
        ax.set_aspect(1)

    plt.tight_layout()
    plt.show()


def _bic(log_likelihood, k, n):
    return -2 * log_likelihood + k * np.log(n)


def _confusion_matrix(task, all_operator_classes, all_parameter_fit_bounds, population_size):
    # Number of models
    n_models = len(all_operator_classes)

    # Data container
    confusion_matrix = np.zeros((n_models, n_models))

    # Set up progress bar
    with tqdm(total=population_size*n_models**2, file=sys.stdout) as pbar:

        # Loop over each model
        for i, m_to_sim in enumerate(all_operator_classes):

            for _ in range(population_size):

                parameters_for_sim = all_parameter_fit_bounds[i]

                # Generate a random agent
                random_agent = None
                if not len(parameters_for_sim) > 0:
                    random_agent = m_to_sim()
                else:
                    random_parameters = _random_parameters(
                        parameter_fit_bounds=parameters_for_sim)
                    random_agent = m_to_sim(**random_parameters)

                # Simulate the task
                simulated_data = _simulate(task=task, operator=random_agent)

                # Container for BIC scores
                bs_scores = np.zeros(n_models)

                # For each model
                for k, m_to_fit in enumerate(all_operator_classes):

                    parameters_for_fit = all_parameter_fit_bounds[k]

                    # Determine best-fit parameter values
                    _, best_fit_objective_value = _best_fit_parameters(
                        operator_class=m_to_fit, parameter_fit_bounds=parameters_for_fit, data=simulated_data, task=task)

                    # Get log-likelihood for best param
                    ll = -best_fit_objective_value

                    # Compute the BIC score
                    n_param_m_to_fit = len(parameters_for_fit)
                    bs_scores[k] = _bic(
                        log_likelihood=ll, k=n_param_m_to_fit, n=len(simulated_data))

                    # Update progress bar
                    pbar.update(1)

                # Get minimum value for bic (min => best)
                min_score = np.min(bs_scores)

                # Get index(es) of models that get best bic
                idx_min = np.flatnonzero(bs_scores == min_score)

                # Add result in matrix
                confusion_matrix[i, idx_min] += 1/len(idx_min)

    # Get the model names
    model_names = [m.__name__ for m in all_operator_classes]

    # Create dataframe
    confusion = pd.DataFrame(confusion_matrix,
                             index=model_names,
                             columns=model_names)

    return confusion


def _plot_confusion_matrix(data):
    # Create figure and axes
    _, ax = plt.subplots(figsize=(12, 10))

    # Display the results using a heatmap
    sns.heatmap(data=data, cmap='viridis', annot=True, ax=ax)

    # Set x-axis and y-axis labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.show()


def _recall(model_name, data):
    # Get the number of true positive
    k = data.at[model_name, model_name]

    # Get the number of true positive + false NEGATIVE
    n = np.sum(data.loc[model_name])

    # Compute the recall and return it
    recall = k/n

    # Compute the confidence interval
    ci_recall = statsmodels.stats.proportion.proportion_confint(
        count=k, nobs=n)

    return recall, ci_recall


def _precision(model_name, data):
    # Get the number of true positive
    k = data.at[model_name, model_name]

    # Get the number of true positive + false POSITIVE
    n = np.sum(data[model_name])

    # Compute the precision
    precision = k/n

    # Compute the confidence intervals
    ci_pres = statsmodels.stats.proportion.proportion_confint(k, n)

    return precision, ci_pres


def _f1(precision, recall):
    # Compute the f score
    f_score = 2*(precision * recall)/(precision+recall)
    return f_score


def _robustness_statistics(all_operator_classes, data):
    # Get the model names
    model_names = [m.__name__ for m in all_operator_classes]

    # Results container
    row_list = []

    # For each model...
    for m in model_names:

        # Compute the recall
        recall, ci_recall = _recall(model_name=m, data=data)

        # Compute the precision and confidence intervals
        precision, ci_pres = _precision(model_name=m, data=data)

        # Compute the f score
        f_score = _f1(precision=precision, recall=recall)

        # Backup
        row_list.append({
            "model": m,
            "Recall": recall,
            "Recall [CI]": ci_recall,
            "Precision": precision,
            "Precision [CI]": ci_pres,
            "F1 score": f_score
        })

    # Create dataframe and display it
    stats = pd.DataFrame(row_list, index=model_names)

    return stats
