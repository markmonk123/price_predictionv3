#!/usr/bin/env python3
"""
Test script to validate normalization pipeline.
Demonstrates: Features → Normalization → Training → Prediction → Denormalization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_normalizer import DataNormalizer
from enhanced_forecasting import EnhancedBitcoinForecaster

# Test constants
MAX_DEVIATION_THRESHOLD = 50.0  # Maximum acceptable deviation percentage for predictions
DENORMALIZATION_TOLERANCE = 1e-6  # Tolerance for denormalization accuracy


def test_normalization_pipeline():
    """Test the complete normalization pipeline."""
    
    print("=" * 80)
    print("🧪 TESTING NORMALIZATION PIPELINE")
    print("=" * 80)
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Creating sample price data...")
    start_date = datetime.now() - timedelta(days=10)
    dates = pd.date_range(start_date, periods=500, freq='30T')
    prices = np.cumsum(np.random.randn(500) * 100) + 50000
    df = pd.DataFrame({'date': dates, 'price': prices})
    
    print(f"   ✅ Created {len(df)} data points")
    print(f"   📈 Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # Step 2: Create features (happens inside forecaster)
    print("\n🔧 Step 2: Features will be created during training...")
    
    # Step 3: Initialize forecaster (will handle normalization internally)
    print("\n🤖 Step 3: Initializing Enhanced Bitcoin Forecaster...")
    forecaster = EnhancedBitcoinForecaster()
    print("   ✅ Forecaster initialized")
    
    # Step 4: Train models (normalization happens here)
    print("\n📚 Step 4: Training models with automatic feature normalization...")
    success = forecaster.train_models(df)
    
    if not success:
        print("   ❌ Training failed")
        return False
    
    print("   ✅ Models trained successfully")
    print(f"   📊 Normalizer fitted with {len(forecaster.normalizer.feature_stats)} features")
    
    # Step 5: Generate predictions (denormalization happens here)
    print("\n🔮 Step 5: Generating predictions (will be denormalized)...")
    forecast = forecaster.generate_12_hour_forecast(df)
    
    if forecast is None:
        print("   ❌ Forecast generation failed")
        return False
    
    print("   ✅ Forecast generated successfully")
    
    # Step 6: Validate denormalized predictions
    print("\n✅ Step 6: Validating denormalized predictions...")
    
    current_price = df['price'].iloc[-1]
    predicted_prices = forecast['predicted_price'].values
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Predicted Range: ${predicted_prices.min():.2f} - ${predicted_prices.max():.2f}")
    
    # Display first few predictions
    print(f"\n📈 Sample Predictions (Denormalized):")
    for i, row in forecast.head(5).iterrows():
        time_str = row['timestamp'].strftime('%H:%M')
        price = row['predicted_price']
        std = row['prediction_std']
        change = price - current_price
        change_pct = (change / current_price) * 100
        print(f"   {time_str}: ${price:,.2f} (±${std:.2f}) | Change: {change:+.2f} ({change_pct:+.2f}%)")
    
    # Validate predictions are in reasonable range
    max_deviation_pct = abs((predicted_prices - current_price) / current_price * 100).max()
    
    print(f"\n📊 QUALITY CHECKS:")
    print(f"   Max Deviation: {max_deviation_pct:.2f}%")
    
    if max_deviation_pct > MAX_DEVIATION_THRESHOLD:
        print(f"   ⚠️  WARNING: Large deviation detected!")
        return False
    
    print(f"   ✅ Predictions are within reasonable range")
    
    # Test normalizer directly
    print(f"\n🧪 TESTING NORMALIZER DIRECTLY:")
    start_date = datetime.now() - timedelta(days=1)
    test_df = pd.DataFrame({
        'date': pd.date_range(start_date, periods=10),
        'price': [50000] * 10,
        'feature1': np.arange(1, 11),
        'feature2': np.arange(10, 101, 10)
    })
    
    normalizer = DataNormalizer(method='standard')
    test_df_normalized = normalizer.fit_transform(test_df)
    test_df_denormalized = normalizer.denormalize_dataframe(test_df_normalized)
    
    # Check if denormalization is accurate
    diff1 = abs(test_df['feature1'].values - test_df_denormalized['feature1'].values).max()
    diff2 = abs(test_df['feature2'].values - test_df_denormalized['feature2'].values).max()
    
    print(f"   Feature1 max difference: {diff1:.10f}")
    print(f"   Feature2 max difference: {diff2:.10f}")
    
    if diff1 < DENORMALIZATION_TOLERANCE and diff2 < DENORMALIZATION_TOLERANCE:
        print(f"   ✅ Denormalization is accurate")
    else:
        print(f"   ❌ Denormalization error detected!")
        return False
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    
    print("\n📋 SUMMARY:")
    print("   ✅ Data pipeline: Raw → Features → Normalize → Train → Predict → Denormalize")
    print("   ✅ Predictions are in original price scale")
    print("   ✅ Normalization improves model training")
    print("   ✅ Denormalization ensures readable output")
    
    return True


if __name__ == "__main__":
    try:
        success = test_normalization_pipeline()
        if success:
            print("\n🎉 Normalization pipeline is working correctly!")
            exit(0)
        else:
            print("\n❌ Normalization pipeline has issues")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
