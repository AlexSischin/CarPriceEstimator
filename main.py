import matplotlib.pyplot as plt
import pandas as pd

from feature import FeatureBuilder

pd.options.display.max_columns = None

resource_file = 'resources/car_data.csv'


def main():
    df = pd.read_csv(resource_file)

    feature_builder = FeatureBuilder(df)
    fdf = feature_builder.build(df)

    print(fdf.nunique())
    print(fdf.corr(numeric_only=True))

    # fdf.plot.scatter(x=F_MAX_POWER, y=F_MODEL_M_PRICE, rot=90)
    plt.show()


if __name__ == '__main__':
    main()
