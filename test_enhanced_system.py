#!/usr/bin/env python3
"""
Test script for the enhanced price prediction system with denormalized data
and progressive training capabilities.
"""

import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_data_preprocessor():
    """Test the data preprocessor module."""
    print("🧪 Testing Data Preprocessor...")
    
    try:
        from data_preprocessor import DataPreprocessor, preprocess_data_for_model
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        prices = np.random.randn(len(dates)).cumsum() + 50000
        
        test_df = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        
        # Test preprocessor
        preprocessor = DataPreprocessor()
        processed_df, metadata = preprocessor.prepare_denormalized_data(test_df)
        
        print(f"   ✅ Processed {processed_df.shape[0]} samples with {len(metadata['feature_columns'])} features")
        print(f"   ✅ Data denormalized: {metadata['denormalized']}")
        print(f"   ✅ Structure preserved: {metadata['structure_preserved']}")
        
        # Test data integrity
        integrity_check = preprocessor.validate_data_integrity(processed_df)
        print(f"   ✅ Data integrity: {'Passed' if integrity_check['validation_passed'] else 'Failed'}")
        
        return True, processed_df
        
    except Exception as e:
        print(f"   ❌ Data preprocessor test failed: {e}")
        return False, None

def test_gpu_optimizer():
    """Test the GPU optimizer module."""
    print("🧪 Testing GPU Optimizer...")
    
    try:
        from gpu_optimizer import GPUOptimizer, setup_gpu_optimization
        
        optimizer = GPUOptimizer()
        results = optimizer.run_full_optimization()
        
        print(f"   ✅ System: {results['system_info']['platform']}")
        print(f"   ✅ GPUs found: {len(results['gpu_info']['gpus'])}")
        print(f"   ✅ CUDA available: {results['gpu_info']['cuda_available']}")
        print(f"   ✅ Optimization complete: {results['optimization_complete']}")
        
        return True, results
        
    except Exception as e:
        print(f"   ❌ GPU optimizer test failed: {e}")
        return False, None

def test_progressive_training():
    """Test the progressive training system."""
    print("🧪 Testing Progressive Training System...")
    
    try:
        from continuous_training import ProgressiveTrainingSystem
        
        # Create sample initial data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5T')  # 5-minute intervals
        prices = np.random.randn(len(dates)).cumsum() + 50000
        
        initial_data = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        
        # Initialize progressive training system
        system = ProgressiveTrainingSystem(update_interval=10, min_models=2, enable_gpu=True)  # Short interval for testing
        
        print("   🚀 Starting progressive training system...")
        system.start_progressive_system(initial_data)
        
        # Run for a short time
        print("   ⏳ Running for 30 seconds...")
        time.sleep(30)
        
        # Get metrics
        metrics = system.training_metrics
        print(f"   ✅ Data updates: {metrics['data_updates']}")
        print(f"   ✅ Total retrains: {metrics['total_retrains']}")
        print(f"   ✅ Active models: {len(system.model_pool)}")
        
        # Test prediction
        prediction = system.get_progressive_prediction()
        if prediction:
            print(f"   ✅ Progressive prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.3f})")
        
        # Stop system
        system.stop_progressive_system()
        print("   ✅ Progressive training system stopped")
        
        return True, system
        
    except Exception as e:
        print(f"   ❌ Progressive training test failed: {e}")
        return False, None

def test_enhanced_prediction():
    """Test the enhanced prediction system."""
    print("🧪 Testing Enhanced Prediction System...")
    
    try:
        # Mock the data fetching function for testing
        def mock_fetch_bitcoin_futures_data():
            dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
            prices = np.random.randn(len(dates)).cumsum() + 50000
            return pd.DataFrame({'date': dates, 'price': prices})
        
        # Temporarily replace the function
        import priceprediction
        original_fetch = priceprediction.fetch_bitcoin_futures_data
        priceprediction.fetch_bitcoin_futures_data = mock_fetch_bitcoin_futures_data
        
        # Test enhanced prediction
        results = priceprediction.enhanced_bitcoin_prediction_with_denormalized_data()
        
        if results:
            print(f"   ✅ Prediction: {results['prediction_label']}")
            print(f"   ✅ Confidence: {results['confidence']:.1%}")
            print(f"   ✅ Data denormalized: {results['data_denormalized']}")
            print(f"   ✅ GPU optimized: {results['gpu_optimized']}")
            print(f"   ✅ Features used: {results['features_used']}")
        
        # Restore original function
        priceprediction.fetch_bitcoin_futures_data = original_fetch
        
        return True, results
        
    except Exception as e:
        print(f"   ❌ Enhanced prediction test failed: {e}")
        return False, None

def run_comprehensive_test():
    """Run comprehensive tests of all enhanced systems."""
    print("\n🚀 COMPREHENSIVE SYSTEM TEST")
    print("="*60)
    
    test_results = {}
    
    # Test 1: Data Preprocessor
    print("\n1️⃣ Testing Data Preprocessor...")
    preprocessor_success, preprocessor_data = test_data_preprocessor()
    test_results['preprocessor'] = preprocessor_success
    
    # Test 2: GPU Optimizer
    print("\n2️⃣ Testing GPU Optimizer...")
    gpu_success, gpu_results = test_gpu_optimizer()
    test_results['gpu_optimizer'] = gpu_success
    
    # Test 3: Progressive Training (only if preprocessor works)
    if preprocessor_success:
        print("\n3️⃣ Testing Progressive Training...")
        training_success, training_system = test_progressive_training()
        test_results['progressive_training'] = training_success
    else:
        print("\n3️⃣ Skipping Progressive Training (preprocessor failed)")
        test_results['progressive_training'] = False
    
    # Test 4: Enhanced Prediction
    print("\n4️⃣ Testing Enhanced Prediction...")
    prediction_success, prediction_results = test_enhanced_prediction()
    test_results['enhanced_prediction'] = prediction_success
    
    # Summary
    print("\n📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return test_results

if __name__ == "__main__":
    print("🧪 Enhanced Price Prediction System - Test Suite")
    print("="*60)
    
    # Check dependencies first
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing core dependencies: {e}")
        sys.exit(1)
    
    # Run comprehensive tests
    results = run_comprehensive_test()
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n✅ All systems operational!")
        sys.exit(0)
    else:
        print("\n❌ Some systems failed!")
        sys.exit(1)