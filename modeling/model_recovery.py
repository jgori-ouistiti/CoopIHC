from .utils import _confusion_matrix, _robustness_statistics, _plot_confusion_matrix


def robustness(task, all_operator_classes, all_parameter_fit_bounds, population_size=20, plot=False):
    confusion_matrix = _confusion_matrix(task=task, all_operator_classes=all_operator_classes,
                                         all_parameter_fit_bounds=all_parameter_fit_bounds, population_size=population_size)

    if plot:
        _plot_confusion_matrix(data=confusion_matrix)

    robustness = _robustness_statistics(
        all_operator_classes=all_operator_classes, data=confusion_matrix)

    return robustness
