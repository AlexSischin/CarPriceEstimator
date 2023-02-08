import re
from functools import reduce

from pandas import DataFrame, Series

from converter import CMVMapper, BCDMapper

# Raw data columns
COL_MODEL = 'Model'
COL_PRICE = 'Price'
COL_YEAR = 'Year'
COL_KILOMETER = 'Kilometer'
COL_FUEL_TYPE = 'Fuel Type'
COL_TRANSMISSION = 'Transmission'
COL_LOCATION = 'Location'
COL_COLOR = 'Color'
COL_OWNER = 'Owner'
COL_SELLER_TYPE = 'Seller Type'
COL_ENGINE = 'Engine'
COL_MAX_POWER = 'Max Power'
COL_MAX_TORQUE = 'Max Torque'
COL_DRIVETRAIN = 'Drivetrain'
COL_LENGTH = 'Length'
COL_WIDTH = 'Width'
COL_HEIGHT = 'Height'
COL_SEATING_CAPACITY = 'Seating Capacity'
COL_FUEL_CAPACITY = 'Fuel Tank Capacity'

# Model features and target
F_MODEL_M_PRICE = 'model_m_price'
F_YEAR = 'year'
F_KILOMETER = 'kilometer'
F_FUEL_TYPE_M_PRICE = 'fuel_type_m_price'
F_TRANSMISSION_AUTO = 'transmission_auto'
F_LOCATION_M_PRICE = 'location_m_price'
F_COLOR_M_PRICE = 'color_m_price'
F_OWNER_M_PRICE = 'owner_m_price'
F_SELLER_M_PRICE = 'seller_m_price'
F_ENGINE_DISP = 'engine_disp'
F_MAX_POWER = 'max_power'
F_MAX_POWER_RPM = 'max_power_rpm'
F_MAX_TORQUE = 'max_torque'
F_MAX_TORQUE_RPM = 'max_torque_rpm'
F_DRIVETRAIN_M_PRICE = 'drivetrain_m_price'
F_LENGTH = 'length'
F_WIDTH = 'width'
F_AREA = 'area'
F_HEIGHT = 'height'
F_PARABOLIC_HEIGHT = 'parabolic_height'
F_SEATING_CAPACITY = 'seating_capacity'
F_FUEL_CAPACITY = 'fuel_capacity'
T_PRICE = 'price'

VAL_TRANSMISSION_AUTO = 'Automatic'
VAL_TRANSMISSION_MANUAL = 'Manual'
PATTERN_ENGINE = re.compile(r'^(\d+)\s*cc$')
PATTERN_POWER = re.compile(r'^([\d.]+)\s*(?:bhp)?\s*@?\s*([\d.]+)\s*(?:rpm)?$', re.RegexFlag.IGNORECASE)
PATTERN_TORQUE = re.compile(r'^([\d.]+)\s*(?:nm)?\s*@?\s*([\d.]+)\s*(?:rpm)?$', re.RegexFlag.IGNORECASE)


def validate_car_df(car_df: DataFrame):
    duplicates = ~car_df.duplicated(keep='first')

    model = car_df[COL_MODEL].notna()
    price = car_df[COL_PRICE].notna()
    year = car_df[COL_YEAR].notna()
    kilometer = car_df[COL_KILOMETER].notna()
    fuel_type = car_df[COL_FUEL_TYPE].notna()
    transmission = car_df[COL_TRANSMISSION].isin([VAL_TRANSMISSION_AUTO, VAL_TRANSMISSION_MANUAL])
    location = car_df[COL_LOCATION].notna()
    color = car_df[COL_COLOR].notna()
    owner = car_df[COL_OWNER].notna()
    seller_type = car_df[COL_SELLER_TYPE].notna()
    engine = car_df[COL_ENGINE].str.extract(PATTERN_ENGINE).notna().all(axis=1)
    max_power = car_df[COL_MAX_POWER].str.extract(PATTERN_POWER).notna().all(axis=1)
    max_torque = car_df[COL_MAX_TORQUE].str.extract(PATTERN_TORQUE).notna().all(axis=1)
    drivetrain = car_df[COL_DRIVETRAIN].notna()
    length = car_df[COL_LENGTH].notna()
    width = car_df[COL_WIDTH].notna()
    height = car_df[COL_HEIGHT].notna()
    seating_capacity = car_df[COL_SEATING_CAPACITY].notna()
    fuel_capacity = car_df[COL_FUEL_CAPACITY].notna()

    valid_rows = reduce(lambda a, b: a & b,
                        [duplicates, model, price, year, kilometer, fuel_type, transmission, location, color, owner,
                         seller_type, engine, max_power, max_torque, drivetrain, length, width, height,
                         seating_capacity, fuel_capacity])
    return valid_rows


class ZScoreNormalizer:
    def __init__(self, feature: Series):
        self._m = feature.mean()
        self._sd = feature.std()

    def scale(self, feature: Series):
        return (feature - self._m) / self._sd


class FeatureBuilder:
    def __init__(self, car_df: DataFrame):
        self._init_converters(car_df)
        raw_fdf = self._build_raw_features(car_df)
        self._init_scalers(raw_fdf)
        self.cached_fdf = self._scale(raw_fdf)

    def build(self, car_df: DataFrame, scale=True) -> DataFrame:
        raw_fdf = self._build_raw_features(car_df)
        return self._scale(raw_fdf) if scale else raw_fdf

    def _init_converters(self, cdf) -> None:
        self._model_mapper = CMVMapper(cdf[COL_MODEL], cdf[COL_PRICE])
        self._fuel_type_mapper = CMVMapper(cdf[COL_FUEL_TYPE], cdf[COL_PRICE])
        self._transmission_mapper = BCDMapper(VAL_TRANSMISSION_AUTO, VAL_TRANSMISSION_MANUAL)
        self._location_mapper = CMVMapper(cdf[COL_LOCATION], cdf[COL_PRICE])
        self._color_mapper = CMVMapper(cdf[COL_COLOR], cdf[COL_PRICE])
        self._owner_mapper = CMVMapper(cdf[COL_OWNER], cdf[COL_PRICE])
        self._seller_type_mapper = CMVMapper(cdf[COL_SELLER_TYPE], cdf[COL_PRICE])
        self._drivetrain_mapper = CMVMapper(cdf[COL_DRIVETRAIN], cdf[COL_PRICE])
        self._mean_height = cdf[COL_HEIGHT].mean()

    def _build_raw_features(self, car_df) -> DataFrame:
        fdf = DataFrame()
        fdf[F_MODEL_M_PRICE] = self._model_mapper.map(car_df[COL_MODEL])
        fdf[F_YEAR] = car_df[COL_YEAR]
        fdf[F_KILOMETER] = car_df[COL_KILOMETER]
        fdf[F_TRANSMISSION_AUTO] = self._transmission_mapper.map(car_df[COL_TRANSMISSION])
        fdf[F_FUEL_TYPE_M_PRICE] = self._fuel_type_mapper.map(car_df[COL_FUEL_TYPE])
        fdf[F_LOCATION_M_PRICE] = self._location_mapper.map(car_df[COL_LOCATION])
        fdf[F_COLOR_M_PRICE] = self._color_mapper.map(car_df[COL_COLOR])
        fdf[F_OWNER_M_PRICE] = self._owner_mapper.map(car_df[COL_OWNER])
        fdf[F_SELLER_M_PRICE] = self._seller_type_mapper.map(car_df[COL_SELLER_TYPE])
        fdf[F_ENGINE_DISP] = car_df[COL_ENGINE].str.extract(PATTERN_ENGINE)
        fdf[F_MAX_POWER] = car_df[COL_MAX_POWER].str.extract(PATTERN_POWER)[0]
        fdf[F_MAX_POWER_RPM] = car_df[COL_MAX_POWER].str.extract(PATTERN_POWER)[1]
        fdf[F_MAX_TORQUE] = car_df[COL_MAX_TORQUE].str.extract(PATTERN_TORQUE)[0]
        fdf[F_MAX_TORQUE_RPM] = car_df[COL_MAX_TORQUE].str.extract(PATTERN_TORQUE)[1]
        fdf[F_DRIVETRAIN_M_PRICE] = self._drivetrain_mapper.map(car_df[COL_DRIVETRAIN])
        fdf[F_LENGTH] = car_df[COL_LENGTH]
        fdf[F_WIDTH] = car_df[COL_WIDTH]
        fdf[F_AREA] = car_df[COL_LENGTH] * car_df[COL_WIDTH]
        fdf[F_HEIGHT] = car_df[COL_HEIGHT]
        fdf[F_PARABOLIC_HEIGHT] = (car_df[COL_HEIGHT] - self._mean_height) ** 2
        fdf[F_SEATING_CAPACITY] = car_df[COL_SEATING_CAPACITY]
        fdf[F_FUEL_CAPACITY] = car_df[COL_FUEL_CAPACITY]
        return fdf.astype(float)

    def _init_scalers(self, raw_fdf):
        self._scalers = {column: ZScoreNormalizer(raw_fdf[column]) for column in raw_fdf}

    def _scale(self, fdf) -> DataFrame:
        for column in fdf:
            fdf[column] = self._scalers[column].scale(fdf[column])
        return fdf
