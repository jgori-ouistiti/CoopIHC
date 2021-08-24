# %%
from envs import MultiBanditTask
from operators import RW, WSLS, RandomOperator

from modeling.parameter_recovery import correlations
from modeling.model_recovery import robustness

from tabulate import tabulate
from collections import OrderedDict

from loguru import logger
logger.remove()

# %%
# Task parameters definition
N = 2
P = [0.5, 0.75]
T = 100

# Task definition
multi_bandit_task = MultiBanditTask(N=N, P=P, T=T)

# Parameter recovery correlations for operator
rw_parameter_fit_bounds = OrderedDict()
rw_parameter_fit_bounds["q_alpha"] = (0., 1.)
rw_parameter_fit_bounds["q_beta"] = (0., 20.)

wsls_parameter_fit_bounds = OrderedDict()
wsls_parameter_fit_bounds["epsilon"] = (0., 1.)

# Population size
N_SIMULATIONS = 20

# %%
# Parameter Recovery: WSLS
print("## Parameter Recovery: WSLS")
correlations_wsls = correlations(task=multi_bandit_task, operator_class=WSLS,
                                 parameter_fit_bounds=wsls_parameter_fit_bounds, population_size=N_SIMULATIONS, plot=True)

print(tabulate(correlations_wsls, headers="keys",
               tablefmt="psql", showindex=False))
print()

# %%
# Parameter Recovery: RW
print("## Parameter Recovery: RW")
correlations_rw = correlations(task=multi_bandit_task, operator_class=RW,
                               parameter_fit_bounds=rw_parameter_fit_bounds, population_size=N_SIMULATIONS, plot=True)
print(tabulate(correlations_rw, headers="keys",
               tablefmt="psql", showindex=False))

# %%
# Model Recovery
print("## Model Recovery")
model_recovery_stats = robustness(task=multi_bandit_task, all_operator_classes=[RandomOperator, WSLS, RW], all_parameter_fit_bounds=[
    [], wsls_parameter_fit_bounds, rw_parameter_fit_bounds], population_size=N_SIMULATIONS, plot=True)
print(tabulate(model_recovery_stats, headers="keys",
               tablefmt="psql", showindex=False))
