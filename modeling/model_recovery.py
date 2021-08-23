import pandas as pd


def robustness(*args, **kwargs):
    data = [{"model": "ExampleModel",
            "Recall": 0.98,
             "Recall [CI]": (0.95, 0.99),
             "Precision": 0.96,
             "Precision [CI]": (0.95, 0.99),
             "F1 score": 0.98}]
    return pd.DataFrame(data)
