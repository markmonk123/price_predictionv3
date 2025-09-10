#!/usr/bin/env python3
"""
Enhanced Bitcoin Price Prediction Demo
Demonstrates the enhanced 12-hour forecasting system with all features.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def run_enhanced_demo():
    """Run a demonstration of the enhanced forecasting system."""
    
    print("🚀 Enhanced Bitcoin Price Prediction System Demo")
    print("="*60)
    
    try:
        # Import the main prediction module
        import priceprediction
        
        print("📊 Running enhanced prediction system...")
        print("⏰ This will demonstrate:")
        print("   • 12-hour price predictions (30-min intervals)")
        print("   • Market sentiment analysis (6-hour window)")
        print("   • Statistical analysis (high/low/median/average)")
        print("   • Multi-model training (5 models)")
        print("   • Continuous training demonstration")
        print("   • Data integrity monitoring")
        print("\n" + "="*60)
        
        # Run the main prediction system
        results = priceprediction.main()
        
        print("\n✅ Demo completed successfully!")
        print("\n📋 Summary of Features Demonstrated:")
        print("   ✅ Fixed data handling issues")
        print("   ✅ 12-hour forecasting with 24 predictions")
        print("   ✅ Market sentiment analysis (BULLISH/BEARISH/NEUTRAL)")
        print("   ✅ Statistical analysis with delta values")
        print("   ✅ Multi-model ensemble (5 models)")
        print("   ✅ Continuous training system")
        print("   ✅ Enhanced output formatting")
        
        return results
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎯 Starting Enhanced Bitcoin Price Prediction Demo...")
    run_enhanced_demo()