# General imports
from matplotlib import pyplot as plt
import matplotlib.axes
import pyro
import seaborn as sns
import pandas as pd

# Import task and users
from bandit.envs import MultiBanditTask
from bandit.agents import WSLS

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
wsls = WSLS(epsilon=0.26)

# Parameter priors for users
wsls_parameter_priors = {
    "epsilon": {"prior": pyro.distributions.Uniform(0.0, 1.0), "bounds": (0.0, 1.0)}
}

# Population size
N_SIMULATIONS = 3


def test_bayesian_parameter_recovery():
    # Define bundle for model checks
    wsls_bundle = ModelChecks(task=multi_bandit_task, user=wsls)

    # Check if WSLS can recover parameters
    bayesian_parameter_recovery_result = wsls_bundle.test_bayesian_parameter_recovery(
        parameter_priors=wsls_parameter_priors,
        seed=RANDOM_SEED,
        num_mcmc_samples=5,
        n_simulations=N_SIMULATIONS,
    )

    # Save MCMC samples DataFrame
    all_mcmc_samples = [
        {
            "Parameter": param_name,
            "Used to simulate": true_param_value,
            "Recovered": mcmc_sample,
        }
        for simulation in bayesian_parameter_recovery_result.simulations
        for param_name, true_param_value in simulation.true_parameters.items()
        for mcmc_sample in {
            k: v.numpy() for k, v in simulation.mcmc.get_samples().items()
        }[param_name]
    ]
    mcmc_samples_df = pd.DataFrame(all_mcmc_samples)
    mcmc_samples_df.to_csv("test/data/mcmc_samples.csv")

    # Check if parameter recovery was successful
    assert hasattr(bayesian_parameter_recovery_result, "success")
    assert bayesian_parameter_recovery_result.success

    # Check if result has stored number of simulations
    assert hasattr(bayesian_parameter_recovery_result, "n_simulations")
    assert bayesian_parameter_recovery_result.n_simulations == N_SIMULATIONS

    # Check if result has stored the parameter priors
    assert hasattr(bayesian_parameter_recovery_result, "parameter_priors")
    assert bayesian_parameter_recovery_result.parameter_priors == wsls_parameter_priors

    # Check if result has attribute for simulations
    assert hasattr(bayesian_parameter_recovery_result, "simulations")
    assert type(bayesian_parameter_recovery_result.simulations) is list
    assert len(bayesian_parameter_recovery_result.simulations) == N_SIMULATIONS
    for simulation in bayesian_parameter_recovery_result.simulations:
        assert hasattr(simulation, "id")
        assert type(simulation.id) is int

        # Check recovery result's user
        assert hasattr(simulation, "user")
        assert type(simulation.user) is WSLS

        # Check recovery result's MCMC
        assert hasattr(simulation, "mcmc")
        assert hasattr(simulation.mcmc, "get_samples")
        mcmc_samples = {k: v.numpy() for k, v in simulation.mcmc.get_samples().items()}
        assert mcmc_samples != {}
        assert "epsilon" in mcmc_samples

        # Check recovery result's simulated data
        assert hasattr(simulation, "data")
        for attr in ["subject", "time", "action", "reward"]:
            assert hasattr(simulation.data, attr)

        # Check recovery result's true parameters
        assert hasattr(simulation, "true_parameters")
        assert type(simulation.true_parameters) is dict
        for param_name in wsls_parameter_priors.keys():
            assert param_name in simulation.true_parameters
            assert type(simulation.true_parameters[param_name]) is float

        # Check each simulation's plot
        assert hasattr(simulation, "plot")

    # Print and plot results of the first simulation
    bayesian_parameter_recovery_result.simulations[0].mcmc.summary(prob=0.95)

    bayesian_parameter_recovery_result.simulations[0].plot()

    # Check recovery result's plot
    assert hasattr(bayesian_parameter_recovery_result, "plot")

    # Display scatterplot
    bayesian_parameter_recovery_result.plot()


if __name__ == "__main__":
    test_bayesian_parameter_recovery()
