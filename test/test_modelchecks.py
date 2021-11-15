# General imports
import collections
import numpy as np
import pandas as pd
import matplotlib.axes

# Import task and users
from bandit.envs import MultiBanditTask
from bandit.agents import RW, WSLS, RandomPlayer

# Import development helper bundle
from core.bundle import ModelChecks

# Seed for reproduceability of the 'true' parameters
RANDOM_SEED = 12345

# Task parameters definition
N = 2
P = [0.5, 0.75]
T = 100

# Task definition
multi_bandit_task = MultiBanditTask(N=N, P=P, T=T)

# User definitions
wsls = WSLS(epsilon=0.1)
rw = RW(q_alpha=0.1, q_beta=1.0)

# Parameter fit bounds for users
wsls_parameter_fit_bounds = {"epsilon": (0.0, 1.0)}
rw_parameter_fit_bounds = {"q_alpha": (0.0, 1.0), "q_beta": (0.0, 20.0)}

# Population size
N_SIMULATIONS = 20


def test_parameter_recovery():
    # Define bundle for model checks
    wsls_bundle = ModelChecks(task=multi_bandit_task, user=wsls)

    # Define thresholds
    CORRELATION_THRESHOLD = 0.6
    SIGNIFICANCE_LEVEL = 0.1

    # Check if WSLS can recover parameters
    parameter_recovery_result = wsls_bundle.test_parameter_recovery(
        parameter_fit_bounds=wsls_parameter_fit_bounds,
        correlation_threshold=CORRELATION_THRESHOLD,
        significance_level=SIGNIFICANCE_LEVEL,
        n_simulations=N_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    # Check if parameter recovery was successful
    assert parameter_recovery_result.success

    # Check recovery result's correlation data
    assert hasattr(parameter_recovery_result, "correlation_data")
    assert type(parameter_recovery_result.correlation_data) is pd.DataFrame
    assert "Subject" in parameter_recovery_result.correlation_data.columns
    assert "Parameter" in parameter_recovery_result.correlation_data.columns
    assert "Used to simulate" in parameter_recovery_result.correlation_data.columns
    assert "Recovered" in parameter_recovery_result.correlation_data.columns
    assert len(parameter_recovery_result.correlation_data) == N_SIMULATIONS

    # Check recovery result's threshold information
    assert hasattr(parameter_recovery_result, "correlation_treshold")
    assert type(parameter_recovery_result.correlation_threshold) is float
    assert parameter_recovery_result.correlation_threshold == CORRELATION_THRESHOLD

    assert hasattr(parameter_recovery_result, "significance_level")
    assert type(parameter_recovery_result.significance_level) is float
    assert parameter_recovery_result.significance_level == SIGNIFICANCE_LEVEL

    # Check recovery result's number of simulations information
    assert hasattr(parameter_recovery_result, "n_simulations")
    assert type(parameter_recovery_result.n_simulations) is int
    assert parameter_recovery_result.n_simulations == N_SIMULATIONS

    # Check recovery result's correlation statistics
    assert hasattr(parameter_recovery_result, "correlation_statistics")
    assert type(parameter_recovery_result.correlation_statistics) is pd.DataFrame
    assert "parameter" in parameter_recovery_result.correlation_statistics.columns
    assert "r" in parameter_recovery_result.correlation_statistics.columns
    assert "p" in parameter_recovery_result.correlation_statistics.columns
    assert (
        f"r>{parameter_recovery_result.correlation_threshold}"
        in parameter_recovery_result.correlation_statistics.columns
    )
    assert (
        f"p<{parameter_recovery_result.significance_level}"
        in parameter_recovery_result.correlation_statistics.columns
    )
    parameter_count = len(wsls_parameter_fit_bounds)
    assert len(parameter_recovery_result.correlation_statistics) == parameter_count

    # Check recovery result's success flags
    assert parameter_recovery_result.parameters_can_be_recovered
    assert parameter_recovery_result.recovered_parameters_correlate

    # Check recovery result's plot
    assert hasattr(parameter_recovery_result, "plot")


def test_model_recovery():
    # Define bundle for model checks
    rw_bundle = ModelChecks(task=multi_bandit_task, user=rw)

    # Define competing models
    other_competing_models = [
        {"model": RandomPlayer, "parameter_fit_bounds": {}},
        {"model": WSLS, "parameter_fit_bounds": wsls_parameter_fit_bounds},
    ]

    # Define F1-score threshold
    F1_THRESHOLD = 0.8

    # Check if RW can be recovered when competing with other models
    model_recovery_result = rw_bundle.test_model_recovery(
        other_competing_models=other_competing_models,
        this_parameter_fit_bounds=rw_parameter_fit_bounds,
        f1_threshold=F1_THRESHOLD,
        n_simulations_per_model=N_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    # Check if model recovery was successful
    assert model_recovery_result.success

    # Check recovery result's confusion matrix data
    assert hasattr(model_recovery_result, "confusion_data")
    assert type(model_recovery_result.confusion_data) is pd.DataFrame
    for competitor in other_competing_models + {"model": RW}:
        model_class_name = type(competitor.model).__name__
        assert model_class_name in model_recovery_result.confusion_data.columns
    all_competing_models_count = len(other_competing_models) + 1
    assert len(model_recovery_result.confusion_data) == all_competing_models_count

    # Check recovery result's threshold information
    assert hasattr(model_recovery_result, "f1_threshold")
    assert type(model_recovery_result.f1_threshold) is float
    assert model_recovery_result.f1_threshold == F1_THRESHOLD

    # Check recovery result's number of simulations information
    assert hasattr(model_recovery_result, "n_simulations_per_model")
    assert type(model_recovery_result.n_simulations_per_model) is int
    assert model_recovery_result.n_simulations_per_model == N_SIMULATIONS

    # Check recovery result's robustness statistics
    assert hasattr(model_recovery_result, "robustness_statistics")
    assert type(model_recovery_result.robustness_statistics) is pd.DataFrame
    assert "model" in model_recovery_result.robustness_statistics.columns
    assert "Recall" in model_recovery_result.robustness_statistics.columns
    assert "Recall [CI]" in model_recovery_result.robustness_statistics.columns
    assert "Precision" in model_recovery_result.robustness_statistics.columns
    assert "Precision [CI]" in model_recovery_result.robustness_statistics.columns
    assert "F1 score" in model_recovery_result.robustness_statistics.columns
    assert (
        f"f1>{model_recovery_result.f1_threshold}"
        in model_recovery_result.robustness_statistics.columns
    )
    assert (
        len(model_recovery_result.robustness_statistics) == all_competing_models_count
    )

    # Check recovery result's plot
    assert hasattr(model_recovery_result, "confusion_matrix")


def test_recoverable_parameter_ranges():
    # Define bundle for model checks
    wsls_bundle = ModelChecks(task=multi_bandit_task, user=wsls)

    # Define thresholds
    CORRELATION_THRESHOLD = 0.7
    SIGNIFICANCE_LEVEL = 0.1

    # Define parameter ranges
    wsls_parameter_ranges = {
        "epsilon": np.linspace(0.0, 1.0, num=6),
    }

    # Determine ranges within the parameter fit bounds where the parameters can be recovered
    recoverable_parameter_ranges_result = wsls_bundle.test_recoverable_parameter_ranges(
        parameter_ranges=wsls_parameter_ranges,
        correlation_threshold=CORRELATION_THRESHOLD,
        significance_level=SIGNIFICANCE_LEVEL,
        n_simulations_per_sub_range=N_SIMULATIONS,
        seed=RANDOM_SEED,
    )

    # Check if recoverable parameter ranges could be identified
    assert recoverable_parameter_ranges_result.success

    # Check recovery result's correlation data
    assert hasattr(recoverable_parameter_ranges_result, "correlation_data")
    assert type(recoverable_parameter_ranges_result.correlation_data) is pd.DataFrame
    assert "Subject" in recoverable_parameter_ranges_result.correlation_data.columns
    assert "Parameter" in recoverable_parameter_ranges_result.correlation_data.columns
    assert (
        "Used to simulate"
        in recoverable_parameter_ranges_result.correlation_data.columns
    )
    assert "Recovered" in recoverable_parameter_ranges_result.correlation_data.columns

    # Check recovery result's threshold information
    assert hasattr(recoverable_parameter_ranges_result, "correlation_treshold")
    assert type(recoverable_parameter_ranges_result.correlation_threshold) is float
    assert (
        recoverable_parameter_ranges_result.correlation_threshold
        == CORRELATION_THRESHOLD
    )

    assert hasattr(recoverable_parameter_ranges_result, "significance_level")
    assert type(recoverable_parameter_ranges_result.significance_level) is float
    assert recoverable_parameter_ranges_result.significance_level == SIGNIFICANCE_LEVEL

    assert hasattr(
        recoverable_parameter_ranges_result, "recovered_parameter_correlation_threshold"
    )
    assert (
        recoverable_parameter_ranges_result.recovered_parameter_correlation_threshold
        is None
    )

    # Check recovery result's number of simulations information
    assert hasattr(recoverable_parameter_ranges_result, "n_simulations_per_sub_range")
    assert type(recoverable_parameter_ranges_result.n_simulations_per_sub_range) is int
    assert (
        recoverable_parameter_ranges_result.n_simulations_per_sub_range == N_SIMULATIONS
    )

    # Check recovery result's correlation statistics
    assert hasattr(recoverable_parameter_ranges_result, "correlation_statistics")
    assert (
        type(recoverable_parameter_ranges_result.correlation_statistics) is pd.DataFrame
    )
    assert (
        "parameter"
        in recoverable_parameter_ranges_result.correlation_statistics.columns
    )
    assert "r" in recoverable_parameter_ranges_result.correlation_statistics.columns
    assert "p" in recoverable_parameter_ranges_result.correlation_statistics.columns
    assert (
        f"r>{recoverable_parameter_ranges_result.correlation_threshold}"
        in recoverable_parameter_ranges_result.correlation_statistics.columns
    )
    assert (
        f"p<{recoverable_parameter_ranges_result.significance_level}"
        in recoverable_parameter_ranges_result.correlation_statistics.columns
    )
    parameter_count = len(wsls_parameter_fit_bounds)
    assert (
        len(recoverable_parameter_ranges_result.correlation_statistics)
        == parameter_count
    )

    # Check recovery result's success flags
    assert (
        recoverable_parameter_ranges_result.parameters_can_be_recovered_for_entire_range
    )
    assert recoverable_parameter_ranges_result.recovered_parameters_correlate

    # Check recovery result's plot
    assert hasattr(recoverable_parameter_ranges_result, "plot")

    # Check recovery result's recoverable ranges
    assert hasattr(recoverable_parameter_ranges_result, "recoverable_parameter_ranges")
    assert (
        type(recoverable_parameter_ranges_result.recoverable_parameter_ranges)
        is collections.OrderedDict
    )
    if recoverable_parameter_ranges_result.success:
        for (
            parameter_name,
            recoverable_ranges,
        ) in recoverable_parameter_ranges_result.recoverable_parameter_ranges.items():
            assert parameter_name in wsls_parameter_ranges.keys()
            assert type(recoverable_ranges) is list
            assert len(recoverable_ranges) > 0
            for recoverable_range in recoverable_ranges:
                assert type(recoverable_range) is tuple
                for bound in recoverable_range:
                    assert type(bound) is float


def test_recoverable_ranges_plot():
    import numpy.random

    wsls_parameter_ranges = {
        "epsilon": numpy.linspace(0.0, 1.0, num=6),
    }

    correlation_threshold = 0.7
    significance_level = 0.05
    len_b = 4
    n_sim_per_range = 10

    statistics = []
    data = []
    for b_min in range(len_b):
        r = numpy.random.normal(loc=correlation_threshold, scale=0.05)
        p = numpy.random.normal(loc=significance_level, scale=0.01)
        fit_bounds = (b_min / len_b, (b_min + 1) / len_b)

        statistics.append(
            {
                "parameter": "epsilon",
                "r": r,
                f"r>{correlation_threshold}": r > correlation_threshold,
                "p": p,
                f"p<{significance_level}": p < significance_level,
                "recoverable": (r > correlation_threshold) and (p < significance_level),
                "fit_bounds": str(fit_bounds),
            }
        )

        for subject_index in range(n_sim_per_range):
            sim_param = np.random.uniform(*fit_bounds)

            data.append(
                {
                    "Subject": subject_index + 1,
                    "Parameter": "epsilon",
                    "Used to simulate": sim_param,
                    "Recovered": np.random.normal(loc=sim_param, scale=0.05),
                    "Fit bounds": str(fit_bounds),
                }
            )

    correlation_data = pd.DataFrame(data)
    correlation_statistics = pd.DataFrame(statistics)

    ModelChecks._recoverable_fit_bounds_result_plot(
        ordered_parameter_ranges=wsls_parameter_ranges,
        correlation_data=correlation_data,
        correlation_statistics=correlation_statistics,
    )


if __name__ == "__main__":
    test_recoverable_ranges_plot()
