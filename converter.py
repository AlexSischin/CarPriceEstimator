from numbers import Number

import pandas as pd
from pandas import Series


# Category to mean value per category mapper
class CMVMapper:
    def __init__(self, categories: Series, values: Series, column_name: str | None = None):
        if categories.shape != values.shape:
            raise ValueError('Categories and values must have the same shape')
        if not issubclass(values.dtype.type, Number):
            raise ValueError('Values must be numeric')

        c_name, v_name = categories.name, values.name
        df = pd.concat((categories, values), axis=1)
        df = df.groupby(c_name).mean(numeric_only=True)

        self._name = f'{c_name}_mean_{v_name}' if not column_name else column_name
        self._dict = df.to_dict()[v_name]

    def map(self, categories: Series):
        mapped_categories = categories.map(self._dict)
        mapped_categories.name = self._name
        return mapped_categories

    @property
    def name(self):
        return self._name

    @property
    def dict(self):
        return self._dict


# Binary category to dummy mapper
class BCDMapper:
    def __init__(self, base_value: str, imaginary_value: str, column_name: str | None = None):
        if not column_name:
            column_name = base_value
        self._name = column_name
        self._dict = {base_value: 1, imaginary_value: 0}

    def map(self, categories: Series):
        dummy_series = categories.map(self._dict)
        dummy_series.name = self._name
        return dummy_series

    @property
    def name(self):
        return self._name

    @property
    def dict(self):
        return self._dict
