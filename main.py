import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.dtypes.common import is_numeric_dtype

from feature import FeatureBuilder, COL_PRICE, validate_car_df, COL_MODEL
from model import LinearRegressor

log = logging.getLogger(__name__)


def load_data(resource_file, test_examples=None, filter_out_unknown_models=True) -> tuple[DataFrame, DataFrame | None]:
    car_df = pd.read_csv(resource_file)
    log.info(f'Car raw DF shape: {car_df.shape}')

    valid_rows = validate_car_df(car_df)
    car_df = car_df[valid_rows]
    log.info(f'Car valid DF shape: {car_df.shape}')

    if not test_examples or test_examples < 0:
        test_examples = 0

    rows = car_df.shape[0]
    car_df = car_df.sample(frac=1)
    train_cdf = car_df.head(rows - test_examples)
    test_cdf = car_df.tail(test_examples)
    if filter_out_unknown_models:
        test_cdf = test_cdf[test_cdf[COL_MODEL].isin(train_cdf[COL_MODEL])]

    log.info(f'Car training DF shape: {train_cdf.shape}')
    log.info(f'Car testing DF shape: {test_cdf.shape}')

    return train_cdf, test_cdf


def initialize_w_and_b(features: DataFrame | np.ndarray) -> tuple[np.ndarray, float]:
    n = features.shape[1]
    return np.zeros(n), 0.


def calc_mean_relative_error(y_estimate: np.ndarray, y_actual: np.ndarray):
    errors = np.abs(y_actual - y_estimate) / y_actual
    mean = np.mean(errors)
    std_err = np.std(errors) / np.sqrt(y_estimate.shape)
    return mean, std_err


def _plot_costs(cost_hist):
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(cost_hist) + 1), cost_hist)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')


def _plot_tests(test_car_df: DataFrame, estimates: np.ndarray):
    targets = test_car_df[COL_PRICE].to_numpy()
    test_car_df = test_car_df.drop([COL_PRICE], axis=1)
    columns = test_car_df.columns
    n = len(columns)

    plot_rows = int(np.floor(np.sqrt(n)))
    plot_columns = int(np.ceil(n / plot_rows))
    fig, axes = plt.subplots(plot_rows, plot_columns, sharey='all')

    for i, column in enumerate(columns):
        features = test_car_df[column]
        plot_row, plot_column = i // plot_columns, i % plot_columns
        ax = axes[plot_row, plot_column]
        ax.scatter(features.to_numpy(), targets, color='blue', marker='.', label='actual')
        ax.scatter(features.to_numpy(), estimates, color='orange', marker='.', label='estimated')
        ax.set_title(f'{column}')
        if plot_column == 0:
            ax.set_ylabel('Price')
        if not is_numeric_dtype(features) and features.nunique() > 3:
            ax.axes.xaxis.set_ticklabels([])


def _main():
    train_car_df, test_car_df = load_data('resources/car_data.csv', test_examples=50, filter_out_unknown_models=True)

    feature_builder = FeatureBuilder(train_car_df)

    train_feature_df = feature_builder.cached_fdf
    test_feature_df = feature_builder.build(test_car_df).fillna(0)

    train_x = train_feature_df.to_numpy(dtype=float)
    train_y = train_car_df[COL_PRICE].to_numpy(dtype=float)
    test_x = test_feature_df.to_numpy(dtype=float)
    test_y = test_car_df[COL_PRICE].to_numpy(dtype=float)

    w, b = initialize_w_and_b(train_x)
    iterations = 300
    learning_rate = 1e-1
    regressor = LinearRegressor(w, b)

    initial_cost = regressor.cost(train_x, train_y)
    log.info(f'Initial cost: {initial_cost:g}')

    learning_hist = regressor.fit(train_x, train_y, learning_rate, iterations, debug=True)
    cost_hist = [hp.cost for hp in learning_hist]

    log.info(f'Convergence cost: {cost_hist[-1]:g}')

    test_y_estimates = regressor.predict(test_x)
    mean_relative_error, std_error = calc_mean_relative_error(test_y_estimates, test_y)
    log.info(f'Average relative error: {float(mean_relative_error):.2f} +- {float(std_error):.2f}')

    _plot_costs(cost_hist)
    _plot_tests(test_car_df, test_y_estimates)
    plt.subplots_adjust(left=0.04, right=0.97, top=0.95, bottom=0.05, hspace=0.5)
    plt.show()


def _configurate():
    import sys

    log_format = ':. %(levelname)s: %(msg)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])

    pd.options.display.max_columns = 50
    pd.options.mode.use_inf_as_na = True


if __name__ == '__main__':
    _configurate()
    _main()
