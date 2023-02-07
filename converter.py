from numbers import Number

import pandas as pd
from pandas import Series


# Category mean value converter
class CMVConverter:
    def __init__(self, categories: Series, values: Series, new_name: str | None = None):
        if categories.shape != values.shape:
            raise ValueError('Categories and values must have the same shape')
        if not issubclass(values.dtype.type, Number):
            raise ValueError('Values must be numeric')
        c_name, v_name = categories.name, values.name
        m_name = f'{c_name}_mean_{v_name}' if not new_name else new_name

        df = pd.concat((categories, values), axis=1)
        df = df.groupby(c_name).mean(numeric_only=True)
        cat_dict = df.to_dict()[v_name]
        converted_cat = categories.map(cat_dict)
        converted_cat.name = m_name

        self._dict = cat_dict
        self._converted_cat = converted_cat
        self._mean = values.mean()

    @property
    def categories(self):
        return self._converted_cat

    @property
    def dict(self):
        return self._dict

    @property
    def mean(self):
        return self._mean

    def convert_obj(self, category: object, def_value=None):
        try:
            return self._dict[category]
        except KeyError:
            return def_value
