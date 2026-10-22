# Normal: full retraining + save at end. (Default.)
python enhanced_prediction.py

# Fast: reuse the cached state for up to 72 hours before retraining.
FAST_PATH_MODEL_STATE=1 python enhanced_prediction.py

# Custom staleness tolerance:
FAST_PATH_MODEL_STATE=1 MAX_STATE_AGE_HOURS=72 python enhanced_prediction.py