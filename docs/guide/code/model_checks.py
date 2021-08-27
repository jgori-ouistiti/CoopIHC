# %% [markdown]
# # Testing Model Checks
#
# This is an example notebook that a researcher might use when developing a computational interaction experiment,
# specifically a user model or an operator class.

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Imports

# %%
from bandit.envs import MultiBanditTask
from bandit.operators import RW, WSLS, RandomOperator

from core.bundle import _DevelopOperator

from loguru import logger
logger.remove()

print("Imports complete.")

# %% [markdown]
# ### Task and Operator Definitions

# %%
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

# %%
# Parameter fit bounds for operators
rw_parameter_fit_bounds = {"q_alpha": (0., 1.), "q_beta": (0., 20.)}
wsls_parameter_fit_bounds = {"epsilon": (0., 1.)}

# Population size
N_SIMULATIONS = 20

print("Parameter recovery definitions complete.")

# %% [markdown]
# ## Model Recovery

# %%
# Model Recovery: RW
print("## Model Recovery: RW")

rw_bundle = _DevelopOperator(task=multi_bandit_task, operator=rw)

other_competing_models = [
    {"model": RandomOperator, "parameter_fit_bounds": {}},
    {"model": WSLS, "parameter_fit_bounds": wsls_parameter_fit_bounds},
]

rw_can_be_recovered = rw_bundle.test_model_recovery(
    other_competing_models=other_competing_models, this_parameter_fit_bounds=rw_parameter_fit_bounds, f1_threshold=0.8, n_simulations=N_SIMULATIONS, plot=True)

print(
    f"RW: Model recovery was {'successful' if rw_can_be_recovered else 'unsuccessful'}.")

# %% [markdown]
# ## Parameter Recovery: RW

# %%
# Parameter Recovery: RW
print("## Parameter Recovery: RW")

rw_bundle = _DevelopOperator(task=multi_bandit_task, operator=rw)

rw_can_recover_parameters = rw_bundle.test_parameter_recovery(
    parameter_fit_bounds=rw_parameter_fit_bounds, correlation_threshold=0.6, significance_level=0.1, n_simulations=N_SIMULATIONS, plot=True)

print(
    f"RW: Parameter recovery was {'successful' if rw_can_recover_parameters else 'unsuccessful'}.")

# %% [markdown]
# ## Parameter Recovery: WSLS

# %%
# Parameter Recovery: WSLS
print("## Parameter Recovery: WSLS")

wsls_bundle = _DevelopOperator(task=multi_bandit_task, operator=wsls)

wsls_can_recover_parameters = wsls_bundle.test_parameter_recovery(
    parameter_fit_bounds=wsls_parameter_fit_bounds, correlation_threshold=0.6, significance_level=0.1, n_simulations=N_SIMULATIONS, plot=True)

print(
    f"WSLS: Parameter recovery was {'successful' if wsls_can_recover_parameters else 'unsuccessful'}.")
