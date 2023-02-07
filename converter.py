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


# Binary category dummy converter
class BCDConverter:
    def __init__(self, categories: Series, base_value: str | None = None, new_name: str | None = None):
        unique_cats = categories.unique()
        if len(unique_cats) != 2:
            raise ValueError(f'Category must contain 2 unique values. Got: {categories.nunique()}')
        if base_value is not None and base_value not in unique_cats:
            raise ValueError(f'Base value must be present in series. '
                             f'Base value: {base_value}. Category values: {unique_cats}')
        if base_value is None and new_name:
            raise ValueError('New name param requires base value param')

        dummy_df = pd.get_dummies(categories)

        if not base_value:
            base_value = dummy_df.columns[0]

        dummy_categories = dummy_df[base_value]
        if new_name:
            dummy_categories.name = new_name

        imaginary_value = [cat for cat in unique_cats if cat != base_value][0]

        dummy_dict = {base_value: 1, imaginary_value: 0}
        self._dummy_dict = dummy_dict
        self._dummy_cat = dummy_categories

    @property
    def categories(self):
        return self._dummy_cat

    @property
    def dict(self):
        return self._dummy_dict

    def convert_obj(self, category: object):
        return self._dummy_dict[str(category)]
