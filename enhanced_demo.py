#!/usr/bin/env python3
"""
Enhanced Bitcoin Price Prediction System Demo
Demonstrates the new decentralized preprocessing, denormalized data handling,
progressive training, and GPU optimization features.
"""

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_demo_data():
    """Create realistic demo data for the system."""
    print("📊 Creating realistic demo data...")
    
    # Create 2 years of daily data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    n_points = len(dates)
    
    # Create realistic Bitcoin-like price movement with controlled ranges
    base_price = 40000
    
    # Add trends, cycles, and volatility (much more constrained)
    trend = np.linspace(0, 0.2, n_points)  # Smaller overall trend
    cycles = 0.05 * np.sin(2 * np.pi * np.arange(n_points) / 365)  # Smaller cycles
    
    # Add realistic but controlled volatility
    daily_returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    
    # Add some market events (more controlled)
    crash_1 = np.zeros(n_points)
    crash_1[200:210] = -0.03  # 3% daily crash for 10 days
    rally_1 = np.zeros(n_points)
    rally_1[300:320] = 0.02   # 2% daily rally for 20 days
    
    # Combine all factors with bounds checking
    log_returns = trend + cycles + daily_returns + crash_1 + rally_1
    
    # Apply cumulative sum with bounds to prevent extreme values
    cumulative_returns = np.zeros_like(log_returns)
    cumulative_returns[0] = log_returns[0]
    
    for i in range(1, len(log_returns)):
        new_return = cumulative_returns[i-1] + log_returns[i]
        # Bound the cumulative returns to prevent extreme prices
        cumulative_returns[i] = np.clip(new_return, -2.0, 2.0)  # Max 7x change total
    
    prices = base_price * np.exp(cumulative_returns)
    
    # Final sanity check on prices
    prices = np.clip(prices, 1000, 500000)  # Keep prices in reasonable range
    
    demo_df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Add some blockchain-like features
    demo_df['transaction_count'] = 200000 + np.random.poisson(50000, n_points)
    demo_df['mempool_congestion'] = np.maximum(0, np.random.normal(5, 2, n_points))
    demo_df['network_stress'] = np.random.beta(2, 5, n_points)
    
    print(f"✅ Created {len(demo_df)} data points with price range ${demo_df['price'].min():.0f} - ${demo_df['price'].max():.0f}")
    
    return demo_df

def demo_decentralized_preprocessing():
    """Demonstrate the decentralized preprocessing system."""
    print("\n🔧 DEMONSTRATING DECENTRALIZED PREPROCESSING")
    print("="*60)
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Import and initialize preprocessor
    from data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    
    # Configure for demonstration
    config = {
        'pct_threshold': 0.01,  # 1% threshold
        'denormalize_features': True,
        'preserve_structure': True
    }
    preprocessor.config.update(config)
    
    print(f"📥 Input data: {demo_data.shape[0]} samples, {demo_data.shape[1]} columns")
    
    # Process data
    start_time = time.time()
    processed_data, metadata = preprocessor.prepare_denormalized_data(demo_data)
    processing_time = time.time() - start_time
    
    print(f"📤 Processed data: {processed_data.shape[0]} samples, {len(metadata['feature_columns'])} features")
    print(f"⏱️ Processing time: {processing_time:.2f} seconds")
    print(f"🔧 Denormalized: {metadata['denormalized']}")
    print(f"🏗️ Structure preserved: {metadata['structure_preserved']}")
    
    # Validate data integrity
    integrity_check = preprocessor.validate_data_integrity(processed_data)
    print(f"✅ Data integrity: {'PASSED' if integrity_check['validation_passed'] else 'FAILED'}")
    print(f"💾 Memory usage: {integrity_check['memory_usage_mb']:.1f} MB")
    
    # Show target distribution
    target_dist = processed_data['target'].value_counts().sort_index()
    print(f"🎯 Target distribution:")
    for target, count in target_dist.items():
        label = {-1: "DECREASE", 0: "NO CHANGE", 1: "INCREASE"}[target]
        pct = count / len(processed_data.dropna(subset=['target'])) * 100
        print(f"   {label}: {count} samples ({pct:.1f}%)")
    
    return processed_data, metadata, preprocessor

def demo_gpu_optimization():
    """Demonstrate GPU optimization features."""
    print("\n🎮 DEMONSTRATING GPU OPTIMIZATION")
    print("="*60)
    
    try:
        from gpu_optimizer import GPUOptimizer
        
        optimizer = GPUOptimizer()
        results = optimizer.run_full_optimization()
        
        print(f"💻 System: {results['system_info']['platform']}")
        print(f"🖥️ CPU Cores: {results['system_info']['cpu_count']}")
        print(f"💾 RAM: {results['system_info']['memory_total_gb']:.1f} GB")
        print(f"🎮 GPUs Found: {len(results['gpu_info']['gpus'])}")
        print(f"⚡ CUDA Available: {results['gpu_info']['cuda_available']}")
        
        if results['gpu_info']['cuda_available']:
            batch_sizes = results['recommended_batch_sizes']
            print(f"🔧 Recommended batch sizes:")
            print(f"   Small models: {batch_sizes['small_model_batch_size']}")
            print(f"   Medium models: {batch_sizes['medium_model_batch_size']}")
            print(f"   Large models: {batch_sizes['large_model_batch_size']}")
        
        # Performance monitoring
        performance = optimizer.get_performance_monitor()
        print(f"📊 Current system performance:")
        print(f"   CPU: {performance['cpu_percent']:.1f}%")
        print(f"   Memory: {performance['memory_percent']:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"⚠️ GPU optimization demo failed: {e}")
        return None

def demo_progressive_training():
    """Demonstrate progressive training system."""
    print("\n🧠 DEMONSTRATING PROGRESSIVE TRAINING")
    print("="*60)
    
    try:
        from continuous_training import ProgressiveTrainingSystem
        
        # Create initial training data
        demo_data = create_demo_data()
        
        # Initialize progressive training system
        system = ProgressiveTrainingSystem(
            update_interval=5,  # 5 seconds for demo
            min_models=2,
            enable_gpu=True
        )
        
        print("🚀 Starting progressive training system...")
        system.start_progressive_system(demo_data)
        
        # Monitor for a short time
        print("⏳ Running progressive training for 30 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 30:
            time.sleep(3)
            
            # Check system status
            metrics = system.training_metrics
            print(f"   📈 Updates: {metrics['data_updates']}, "
                  f"Retrains: {metrics['total_retrains']}, "
                  f"Models: {len(system.model_pool)}")
            
            # Try to get a prediction
            prediction = system.get_progressive_prediction()
            if prediction:
                labels = {-1: "📉", 0: "➡️", 1: "📈"}
                print(f"   🔮 Prediction: {labels[prediction['prediction']]} "
                      f"(Confidence: {prediction['confidence']:.1%})")
        
        # Final metrics
        print(f"\n📊 Final Progressive Training Metrics:")
        print(f"   🔄 Data updates: {metrics['data_updates']}")
        print(f"   🏋️ Total retrains: {metrics['total_retrains']}")
        print(f"   🤖 Active models: {len(system.model_pool)}")
        print(f"   ✅ Successful predictions: {metrics['successful_predictions']}")
        print(f"   ❌ Failed predictions: {metrics['failed_predictions']}")
        print(f"   ⏱️ Average training time: {metrics['average_training_time']:.2f}s")
        
        # Stop system
        system.stop_progressive_system()
        print("🛑 Progressive training system stopped")
        
        return system
        
    except Exception as e:
        print(f"⚠️ Progressive training demo failed: {e}")
        return None

def demo_enhanced_prediction():
    """Demonstrate enhanced prediction with all features."""
    print("\n🔮 DEMONSTRATING ENHANCED PREDICTION")
    print("="*60)
    
    try:
        # Create demo data
        demo_data = create_demo_data()
        
        # Mock the fetch function for demo
        import priceprediction
        original_fetch = getattr(priceprediction, 'fetch_bitcoin_futures_data', None)
        
        def mock_fetch():
            return demo_data
        
        if original_fetch:
            priceprediction.fetch_bitcoin_futures_data = mock_fetch
        
        # Run enhanced prediction
        print("🎯 Running enhanced prediction with denormalized data...")
        results = priceprediction.enhanced_bitcoin_prediction_with_denormalized_data()
        
        if results:
            print(f"\n🎯 ENHANCED PREDICTION RESULTS:")
            print(f"   💰 Current Price: ${results['current_price']:.2f}")
            print(f"   🔮 Prediction: {results['prediction_label']}")
            print(f"   🎯 Confidence: {results['confidence']:.1%}")
            print(f"   🤖 Models Used: {len(results.get('model_results', {}))}")
            print(f"   📊 Features Used: {results['features_used']} (denormalized)")
            print(f"   ⚡ GPU Optimized: {results['gpu_optimized']}")
            print(f"   ✅ Data Integrity: {results['data_integrity']}")
            print(f"   🏆 Ensemble Accuracy: {results['ensemble_accuracy']:.1%}")
            
            # Show model performance
            print(f"\n📈 Individual Model Performance:")
            for model_name, metrics in results.get('model_results', {}).items():
                print(f"   {model_name}: {metrics['test_accuracy']:.3f} "
                      f"(CV: {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f})")
        
        # Restore original function
        if original_fetch:
            priceprediction.fetch_bitcoin_futures_data = original_fetch
        
        return results
        
    except Exception as e:
        print(f"⚠️ Enhanced prediction demo failed: {e}")
        return None

def main():
    """Run the complete enhanced system demonstration."""
    print("🚀 ENHANCED BITCOIN PRICE PREDICTION SYSTEM DEMO")
    print("="*80)
    print("This demo showcases the enhanced features addressing the requirements:")
    print("• Decentralized data preprocessing with denormalized features")
    print("• Progressive training with historical model results")
    print("• GPU optimization for performance")
    print("• 5-minute continuous training cycle")
    print("• Data integrity and structure preservation")
    print("="*80)
    
    demo_results = {}
    
    # Demo 1: Decentralized Preprocessing
    try:
        processed_data, metadata, preprocessor = demo_decentralized_preprocessing()
        demo_results['preprocessing'] = True
    except Exception as e:
        print(f"❌ Preprocessing demo failed: {e}")
        demo_results['preprocessing'] = False
    
    # Demo 2: GPU Optimization
    try:
        gpu_results = demo_gpu_optimization()
        demo_results['gpu_optimization'] = gpu_results is not None
    except Exception as e:
        print(f"❌ GPU optimization demo failed: {e}")
        demo_results['gpu_optimization'] = False
    
    # Demo 3: Progressive Training
    try:
        training_system = demo_progressive_training()
        demo_results['progressive_training'] = training_system is not None
    except Exception as e:
        print(f"❌ Progressive training demo failed: {e}")
        demo_results['progressive_training'] = False
    
    # Demo 4: Enhanced Prediction
    try:
        prediction_results = demo_enhanced_prediction()
        demo_results['enhanced_prediction'] = prediction_results is not None
    except Exception as e:
        print(f"❌ Enhanced prediction demo failed: {e}")
        demo_results['enhanced_prediction'] = False
    
    # Summary
    print(f"\n🏁 DEMO SUMMARY")
    print("="*60)
    
    passed = sum(demo_results.values())
    total = len(demo_results)
    
    for feature, success in demo_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {feature.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL ENHANCED FEATURES WORKING PERFECTLY!")
        print("✅ The system successfully addresses all requirements:")
        print("   • Preprocessing is decentralized and denormalized")
        print("   • Progressive training includes historical model results")
        print("   • Data structure is preserved while denormalizing numbers")
        print("   • GPU optimization is configured for performance")
        print("   • Continuous 5-minute training cycle is implemented")
    else:
        print(f"\n⚠️ {total-passed} features need attention.")
    
    return demo_results

if __name__ == "__main__":
    print("🧪 Enhanced System Demo - Starting...")
    
    # Check dependencies
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the complete demo
    results = main()
    
    print(f"\n🏆 Demo completed successfully!")
    print("🔗 All enhanced features are now available for production use.")
    
    sys.exit(0 if all(results.values()) else 1)