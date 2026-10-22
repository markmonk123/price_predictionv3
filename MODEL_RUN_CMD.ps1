# Normal: full retraining + save at end. (Default.)
python enhanced_prediction.py

# Fast: reuse the cached state for up to 72 hours before retraining.
FAST_PATH_MODEL_STATE=1 python enhanced_prediction.py

# Custom staleness tolerance:
FAST_PATH_MODEL_STATE=1 MAX_STATE_AGE_HOURS=72 python enhanced_prediction.py

# imbalanced-learn experiment (WSL/bash): train a new oversampled classifier.
CLASS_BALANCING=random-over FAST_PATH_MODEL_STATE=0 python enhanced_prediction.py

# Comparison: train with the previous balanced sample-weight method.
CLASS_BALANCING=sample-weight FAST_PATH_MODEL_STATE=0 python enhanced_prediction.py
