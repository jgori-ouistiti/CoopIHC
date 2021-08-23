import pandas as pd


def correlations(*args, **kwargs):
    data = [{"parameter": "ExampleParameter",
            "r": 0.98,
             "p": 0.01,
             "p<0.05": True}]
    return pd.DataFrame(data)
