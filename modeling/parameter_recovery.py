from .utils import _likelihood, _pearsons_r, _plot_correlations


def correlations(task, operator_class, parameter_fit_bounds, population_size=20, significance_level=0.05, plot=False):
    likelihood = _likelihood(task=task, operator_class=operator_class,
                             parameter_fit_bounds=parameter_fit_bounds, population_size=population_size)

    if plot:
        _plot_correlations(
            parameter_fit_bounds=parameter_fit_bounds, data=likelihood)

    pearsons_r = _pearsons_r(
        operator_class=operator_class, parameter_fit_bounds=parameter_fit_bounds, data=likelihood, significance_level=significance_level)

    return pearsons_r
