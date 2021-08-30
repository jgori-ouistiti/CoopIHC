# %% [markdown]
# # Testing Model Checks
#
# This is an example notebook / interactive python file that a researcher might use when developing a computational interaction experiment,
# specifically a user model or an operator class. It will import a pre-defined task and multiple operator classes of increasing complexity
# for a Bandit task and explore parameter recovery and model recovery for those models.

# %% [markdown]
# ## Setup
#
# For this example notebook, we consider a scenario where a researcher has developed a task--in this case a Bandit task--as well as several
# user models that they want to test for their quality. In particular, they want to test them for their ability to recover 'true' (i.e. known)
# parameters from data (i.e. parameter recovery) as well as for their ability to be recovered when competing with different alternative models
# for simulated data created with the model (i.e. model recovery).
#
# In order to achieve these goals, we first need to import the necessary task and operator definitions and specify the task as well as the
# recovery parameters.

# %% [markdown]
# ### Imports
#
# We first import the task and operator classes as well as the `_DevelopOperator` class that gives us access to the methods for parameter
# and model recovery.

# %%
# Import task and operators
from bandit.envs import MultiBanditTask
from bandit.operators import RW, WSLS, RandomOperator

# Import development helper bundle
from core.bundle import _DevelopOperator

# Disable additional logging
from loguru import logger
logger.remove()

print("Imports complete.")

# %% [markdown]
# ### Task and Operator Definitions
#
# Next, we define specific task to perform using the respective parameters `N` (i.e. the number of bandits that the user can choose from), `P`
# (i.e. the reward probability for each bandit) and `T` (i.e. the number of trials that the user is allowed to perform).
#
# We additionally initialise the operators that we will test.

# %%
# Seed for reproduceability of the 'true' parameters
RANDOM_SEED = 12345

# Task parameters definition
N = 2
P = [0.5, 0.75]
T = 100

# Task definition
multi_bandit_task = MultiBanditTask(N=N, P=P, T=T)

# Operator definitions
wsls = WSLS(epsilon=0.1)
rw = RW(q_alpha=0.1, q_beta=1.)

print("Task and operator definitions complete.")

# %% [markdown]
# ### Parameter Recovery Definitions
#
# First, we will perform a test for parameter recovery for two of the three overall operator classes `WSLS` and `RW`.
# For this, we first need to define the so-called parameter fit bounds (i.e. the name, minimum and maximum value for each
# model parameter) for each model. These will be used to generate random parameter values which, in turn, will be used to
# generate random artificial agents. These artificial agents we will use to generate behavioral data with known parameter
# values which will be tried to recover. We, therefore, need to additionally define the population size or the number of
# artificial agents to create. Typically, the number of agents would be much higher (e.g. 100 agents or more), but since
# this is just a tutorial, we have chosen a rather small number of agents that will produce results in less time.

# %%
# Parameter fit bounds for operators
rw_parameter_fit_bounds = {"q_alpha": (0., 1.), "q_beta": (0., 20.)}
wsls_parameter_fit_bounds = {"epsilon": (0., 1.)}

# Population size
N_SIMULATIONS = 20

print("Parameter recovery definitions complete.")

# %% [markdown]
# ## Parameter Recovery: WSLS
#
# We will start the model checks with a test for parameter recovery for the Win-Stay-Lose-Swith `WSLS` model. This model
# has just a single parameter `epsilon` which can range from 0.0 to 1.0. For the parameter recovery test, we need to use
# the `_DevelopOperator` bundle as it gives us access to the `test_parameter_recovery` method which creates the specified
# number of agents using random parameter values and tries to recover these known parameters from behavioral data created
# using these random agents. It will return `True` if the known 'true' and the recovered parameter values correlate with
# a Pearson's r coefficient value greater than the specified `correlation_threshold` and a significance score p less than
# the specified `significance_level`. It will further plot a scatter plot of the recovered and 'true' parameters and save
# this plot to the specified path.

# %%
# Parameter Recovery: WSLS
print("## Parameter Recovery: WSLS")

# Define bundle for model checks
wsls_bundle = _DevelopOperator(task=multi_bandit_task, operator=wsls)

# Check if WSLS can recover parameters
wsls_can_recover_parameters = wsls_bundle.test_parameter_recovery(
    parameter_fit_bounds=wsls_parameter_fit_bounds,
    correlation_threshold=0.6,
    significance_level=0.1,
    n_simulations=N_SIMULATIONS,
    plot=True,
    save_plot_to="wsls_parameter_recovery.png",
    seed=RANDOM_SEED)

# Print result
print(
    f"WSLS: Parameter recovery was {'successful' if wsls_can_recover_parameters else 'unsuccessful'}.")

# %% [markdown]
# ## Parameter Recovery: RW
#
# Similarly to the parameter recovery example above, we will test parameter recovery for a model with two parameters next.
# All the general structure remains the same but the output will create a plot and statistic for each parameter separately.
# The result of `test_parameter_recovery` will be `True` if all parameters were recovered meeting the specified thresholds.

# %%
# Parameter Recovery: RW
print("## Parameter Recovery: RW")

# Define bundle for model checks
rw_bundle = _DevelopOperator(task=multi_bandit_task, operator=rw)

# Check if WSLS can recover parameters
rw_can_recover_parameters = rw_bundle.test_parameter_recovery(
    parameter_fit_bounds=rw_parameter_fit_bounds,
    correlation_threshold=0.6,
    significance_level=0.1,
    n_simulations=N_SIMULATIONS,
    plot=True,
    save_plot_to="rw_parameter_recovery.png",
    seed=RANDOM_SEED)

# Print result
print(
    f"RW: Parameter recovery was {'successful' if rw_can_recover_parameters else 'unsuccessful'}.")

# %% [markdown]
# ## Model Recovery
#
# Now, we will turn towards model recovery. This test, which can be used on a `_DevelopOperator` bundle using the `test_model_recovery`
# method, creates the specified number of artificial agents (see Parameter Recovery for procedure) for each of the competing models,
# and executes the task for each agent. This simulated behavioral data is then used to try and recover the known underlying model by
# first deriving the best-fit parameter values for each competing model and then comparing the best-fit models using a metric called
# Bayesian information criterion (BIC-score). The result is a confusion matrix of each 'true' known model used to generate the artificial data
# and the recovered model. This confusion matrix is finally used to calculate precision, recall and F1-score for each of the competing
# models. If, for the specified model, the F1-score meets the specified threshold, `test_model_recovery` returns `True`.

# %%
# Model Recovery: RW
print("## Model Recovery: RW")

# Define bundle for model checks
rw_bundle = _DevelopOperator(task=multi_bandit_task, operator=rw)

# Define competing models
other_competing_models = [
    {"model": RandomOperator, "parameter_fit_bounds": {}},
    {"model": WSLS, "parameter_fit_bounds": wsls_parameter_fit_bounds},
]

# Check if RW can be recovered when competing with other models
rw_can_be_recovered = rw_bundle.test_model_recovery(
    other_competing_models=other_competing_models,
    this_parameter_fit_bounds=rw_parameter_fit_bounds,
    f1_threshold=0.8,
    n_simulations_per_model=N_SIMULATIONS,
    plot=True,
    save_plot_to="model_recovery.png",
    seed=RANDOM_SEED)

# Print result
print(
    f"RW: Model recovery was {'successful' if rw_can_be_recovered else 'unsuccessful'}.")
