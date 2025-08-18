import priceprediction
import traceback

try:
    print("--- Starting execution from test_runner.py ---")
    priceprediction.main()
    print("--- Finished execution from test_runner.py ---")
except Exception as e:
    print("--- An error occurred ---")
    print(f"Error: {e}")
    print(traceback.format_exc())
