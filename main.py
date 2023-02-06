import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_columns = None

resource_file = 'resources/car_data.csv'

COL_MAKE = 'Make'
COL_MODEL = 'Model'
COL_PRICE = 'Price'
COL_YEAR = 'Year'
COL_KILOMETER = 'Kilometer'
COL_FUEL = 'Fuel Type'
COL_TRANSMISSION = 'Transmission'
COL_LOCATION = 'Location'
COL_COLOR = 'Color'
COL_OWNER = 'Owner'
COL_SELLER = 'Seller Type'
COL_ENGINE = 'Engine'
COL_POWER = 'Max Power'
COL_TORQUE = 'Max Torque'
COL_DRIVETRAIN = 'Drivetrain'
COL_LENGTH = 'Length'
COL_WIDTH = 'Width'
COL_HEIGHT = 'Height'
COL_SEATING_CAPACITY = 'Seating Capacity'
COL_FUEL_CAPACITY = 'Fuel Tank Capacity'

engine_pattern = re.compile(r'(\d+)\s*cc')
power_pattern = re.compile(r'^([\d.]+)\s*(?:bhp)?\s*@?\s*([\d.]+)\s*(?:rpm)?$', re.RegexFlag.IGNORECASE)
torque_pattern = re.compile(r'^([\d.]+)\s*(?:nm)?\s*@?\s*([\d.]+)\s*(?:rpm)?$', re.RegexFlag.IGNORECASE)


def parse_max_torque(torque: str):
    return torque_pattern.match(torque).groups()


def parse_engine(engine: str):
    return engine_pattern.match(engine).groups()


# Z-score normalization
class Scaler:
    def __init__(self, feature: np.ndarray):
        self._m = np.mean(feature)
        self._sd = np.std(feature)

    def scale(self, feature):
        return (feature - self._m) / self._sd


# category mean price
def cmp_digitize(df, category_var, dependent_var, new_name=None, drop_old=True):
    category_var_new_name = f'{category_var}_mean_{dependent_var}' if new_name is None else new_name
    cat_mp_df = df[[category_var, dependent_var]].groupby(by=category_var).mean()
    cat_mp_df.rename(columns={dependent_var: category_var_new_name}, inplace=True)
    cat_mp_df.reset_index(inplace=True)
    new_df = df.merge(cat_mp_df, on=category_var, how='left')
    if drop_old:
        new_df.drop(category_var, axis=1, inplace=True)
    return new_df


def dummy_digitize(df, category_var, drop_old=True):
    dummies_df = pd.get_dummies(df[category_var], drop_first=True)
    new_df = pd.concat([df, dummies_df], axis=1)
    if drop_old:
        new_df.drop(category_var, axis=1, inplace=True)
    return new_df


def main():
    df = pd.read_csv(resource_file)
    df = df[df.notna()]
    df.drop_duplicates(inplace=True)

    df.drop(COL_MAKE, axis=1, inplace=True)
    df = cmp_digitize(df, COL_MODEL, COL_PRICE)
    df = cmp_digitize(df, COL_FUEL, COL_PRICE)
    df = dummy_digitize(df, COL_TRANSMISSION)
    df = cmp_digitize(df, COL_LOCATION, COL_PRICE)
    df = cmp_digitize(df, COL_COLOR, COL_PRICE)
    df = cmp_digitize(df, COL_OWNER, COL_PRICE)
    df = cmp_digitize(df, COL_SELLER, COL_PRICE)
    df = cmp_digitize(df, COL_DRIVETRAIN, COL_PRICE)
    df['area'] = df[COL_LENGTH] * df[COL_WIDTH]
    df['parabolic_height'] = (df[COL_HEIGHT] - np.mean(df[COL_HEIGHT])) ** 2

    df[COL_ENGINE] = df[COL_ENGINE].str.extract(engine_pattern).astype(float)

    power = df[COL_POWER].str.extract(power_pattern).astype(float)
    power.columns = ['BHP', 'BHP_RPM']
    df.drop(COL_POWER, axis=1, inplace=True)
    df = pd.concat([df, power], axis=1)

    torque = df[COL_TORQUE].str.extract(torque_pattern).astype(float)
    torque.columns = ['NM', 'NM_RPM']
    df.drop('Max Torque', axis=1, inplace=True)
    df = pd.concat([df, torque], axis=1)

    print(df.nunique())
    print(df.corr(numeric_only=True))

    df.plot.scatter(x='Manual', y=COL_PRICE, rot=90)
    plt.show()

    # df.drop_duplicates(inplace=True)
    #
    # torque = df['Max Torque'].str.extract(torque_pattern)
    # torque.columns = ['NM', 'NM_RPM']
    # df.drop('Max Torque', axis=1, inplace=True)
    # df.join(torque)
    #
    # df = df.astype(float)
    # print(df.info())


if __name__ == '__main__':
    main()
