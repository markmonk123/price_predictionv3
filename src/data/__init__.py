"""
Data utilities package.
"""

from .fetch_crypto_data import (
    fetch_crypto_historical_data,
    engineer_features as fetch_engineer_features,
    create_target as fetch_create_target,
    prepare_crypto_dataset,
    save_crypto_dataset as fetch_save_dataset
)

from .generate_crypto_data import (
    generate_realistic_ohlcv,
    engineer_features as gen_engineer_features,
    create_target as gen_create_target,
    generate_crypto_dataset,
    save_crypto_dataset as generate_save_dataset
)

__all__ = [
    # Real data fetching (requires network access)
    'fetch_crypto_historical_data',
    'prepare_crypto_dataset',
    'fetch_save_dataset',
    # Realistic data generation (offline)
    'generate_realistic_ohlcv',
    'generate_crypto_dataset',
    'generate_save_dataset',
    # Common utilities
    'fetch_engineer_features',
    'gen_engineer_features',
    'fetch_create_target',
    'gen_create_target',
]

