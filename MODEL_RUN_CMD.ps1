# Normal: full retraining + save at end. (Default.)
python enhanced_prediction.py

# Fast: serve the cached state when it's fresher than the threshold.
FAST_PATH_MODEL_STATE=1 python enhanced_prediction.py

# Custom staleness tolerance:
FAST_PATH_MODEL_STATE=1 MAX_STATE_AGE_HOURS=2 python enhanced_prediction.py