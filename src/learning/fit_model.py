from typing import Any, Dict

from sklearn.model_selection import train_test_split

from src.learning.preprocessing import RealEstatePreprocessor, PriceModel

config = {
    'dtypes': { 'listing_id': 'int64', 'platform_id': 'int32', 'area': 'float64',
                'rooms': 'Int32', 'floor': 'Int32', 'house_floors': 'Int32',
                'mortgage_rate': 'float32' },
    'parse_dates': ['published_date', 'updated_date'],
    'array_columns': ['subway_distances', 'subway_names'],
    'numeric_fillna': {'rooms': -1, 'floor': -1, 'house_floors': -1},
    'bool_columns': ['pin_color', 'auction_status', 'placement_paid', 'big_card'],
    'center_latitude': 55.751244, 'center_longitude': 37.618423
}

features = [
    'mortgage_rate', 'area', 'rooms', 'floor',
    'latitude', 'longitude',
    'num_subways', 'subway_min_dist', 'subway_mean_dist',
    'distance_to_center', 'lat_bucket', 'lon_bucket',
    'days_since_published', 'days_since_updated',
    'floor_ratio', 'rooms_density',
    'desc_len', 'desc_word_count'
]
cat_features = ['platform_id', 'distance_bucket', 'primary_subway']
target = 'price_per_sqm'

def run_training_pipeline(data_path: str = 'data.parquet') -> Dict[str, Any]:
    preprocessor = RealEstatePreprocessor(**config)
    df = preprocessor.load_data(data_path)
    df = preprocessor.preprocess(df)
    df = preprocessor.feature_engineer(df)
    X, y = preprocessor.get_features_and_target(df, features, cat_features, target)

    mask = y.notna()
    if not mask.any():
        raise ValueError("нет таргета")
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = PriceModel()
    model.train(X_train, y_train, X_test, y_test, cat_features)
    results = model.evaluate(X_test, y_test)
    model.save('price_per_sqm_model.cbm')
    return results
