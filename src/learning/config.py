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
    'area',
    'rooms',
    'floor',
    'latitude',
    'longitude',
    'num_subways',
    'subway_min_dist',
    'subway_mean_dist',
    'distance_to_center',
    'lat_bucket',
    'lon_bucket',
    'floor_ratio',
    'rooms_density',
    'desc_len',
    'desc_word_count'
]
cat_features = ['platform_id', 'distance_bucket', 'primary_subway']
target = 'price_per_sqm'