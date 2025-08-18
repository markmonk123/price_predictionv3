import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Divider,
  Grid,
  TextField,
  Button,
  FormControl,
  FormControlLabel,
  Switch,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions
} from '@mui/material';
import {
  Settings as SettingsIcon,
  SaveAlt as SaveIcon,
  Notifications as NotificationsIcon,
  Security as SecurityIcon,
  Devices as DevicesIcon,
  BarChart as ChartIcon,
  Language as LanguageIcon,
  Code as CodeIcon,
  DeleteOutline as DeleteIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';

import { reconnectFixSession } from '../services/fixService';

function Settings() {
  // FIX connection settings
  const [fixSettings, setFixSettings] = useState({
    senderCompID: 'BITCOIN_PREDICTION_CLIENT',
    targetCompID: 'EXCHANGE',
    heartbeatInterval: 30,
    reconnectInterval: 5,
    logFixMessages: true
  });

  // Prediction settings
  const [predictionSettings, setPredictionSettings] = useState({
    interval: 60,
    threshold: 0.002,
    modelType: 'ensemble',
    featureCount: 50,
    enableBacktesting: true
  });

  // Notification settings
  const [notificationSettings, setNotificationSettings] = useState({
    enablePredictionAlerts: true,
    enableOrderNotifications: true,
    enablePriceAlerts: false,
    priceAlertThreshold: 0.5,
    enableSoundAlerts: true,
    enableBrowserNotifications: false
  });

  // UI state
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [reconnectDialogOpen, setReconnectDialogOpen] = useState(false);

  // Handle FIX settings change
  const handleFixSettingsChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFixSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  // Handle prediction settings change
  const handlePredictionSettingsChange = (e) => {
    const { name, value, type, checked } = e.target;
    setPredictionSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  // Handle notification settings change
  const handleNotificationSettingsChange = (e) => {
    const { name, value, type, checked } = e.target;
    setNotificationSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  // Handle FIX settings save
  const handleSaveFixSettings = () => {
    // In a real application, this would call an API to save settings
    try {
      setSaveSuccess(true);
      setSaveError('');
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (error) {
      setSaveError('Failed to save FIX settings');
    }
  };

  // Handle prediction settings save
  const handleSavePredictionSettings = () => {
    // In a real application, this would call an API to save settings
    try {
      setSaveSuccess(true);
      setSaveError('');
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (error) {
      setSaveError('Failed to save prediction settings');
    }
  };

  // Handle notification settings save
  const handleSaveNotificationSettings = () => {
    // In a real application, this would call an API to save settings
    try {
      setSaveSuccess(true);
      setSaveError('');
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (error) {
      setSaveError('Failed to save notification settings');
    }
  };

  // Handle FIX session reconnect
  const handleReconnectFixSession = async () => {
    try {
      await reconnectFixSession();
      setReconnectDialogOpen(false);
      setSaveSuccess(true);
      setSaveError('');
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (error) {
      setSaveError('Failed to reconnect FIX session');
      setReconnectDialogOpen(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Settings
      </Typography>

      {saveSuccess && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Settings saved successfully
        </Alert>
      )}

      {saveError && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {saveError}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* FIX Protocol Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <CodeIcon sx={{ mr: 1 }} />
                  FIX Protocol Settings
                </Box>
              }
              action={
                <Button 
                  variant="outlined" 
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={() => setReconnectDialogOpen(true)}
                >
                  Reconnect
                </Button>
              }
            />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Sender CompID"
                    name="senderCompID"
                    value={fixSettings.senderCompID}
                    onChange={handleFixSettingsChange}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Target CompID"
                    name="targetCompID"
                    value={fixSettings.targetCompID}
                    onChange={handleFixSettingsChange}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Heartbeat Interval (seconds)"
                    name="heartbeatInterval"
                    type="number"
                    value={fixSettings.heartbeatInterval}
                    onChange={handleFixSettingsChange}
                    margin="normal"
                    InputProps={{ inputProps: { min: 1 } }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Reconnect Interval (seconds)"
                    name="reconnectInterval"
                    type="number"
                    value={fixSettings.reconnectInterval}
                    onChange={handleFixSettingsChange}
                    margin="normal"
                    InputProps={{ inputProps: { min: 1 } }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={fixSettings.logFixMessages}
                        onChange={handleFixSettingsChange}
                        name="logFixMessages"
                      />
                    }
                    label="Log FIX Messages"
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={handleSaveFixSettings}
                  >
                    Save FIX Settings
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <ChartIcon sx={{ mr: 1 }} />
                  Prediction Model Settings
                </Box>
              }
            />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Prediction Interval (seconds)"
                    name="interval"
                    type="number"
                    value={predictionSettings.interval}
                    onChange={handlePredictionSettingsChange}
                    margin="normal"
                    InputProps={{ inputProps: { min: 10 } }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Price Threshold (%)"
                    name="threshold"
                    type="number"
                    value={predictionSettings.threshold * 100}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value) / 100;
                      setPredictionSettings(prev => ({
                        ...prev,
                        threshold: value
                      }));
                    }}
                    margin="normal"
                    InputProps={{ inputProps: { min: 0.1, step: 0.1 } }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel id="model-type-label">Model Type</InputLabel>
                    <Select
                      labelId="model-type-label"
                      name="modelType"
                      value={predictionSettings.modelType}
                      label="Model Type"
                      onChange={handlePredictionSettingsChange}
                    >
                      <MenuItem value="ensemble">Ensemble (RF, GB, LR, SVM)</MenuItem>
                      <MenuItem value="randomForest">Random Forest</MenuItem>
                      <MenuItem value="gradientBoosting">Gradient Boosting</MenuItem>
                      <MenuItem value="logisticRegression">Logistic Regression</MenuItem>
                      <MenuItem value="svm">Support Vector Machine</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Feature Count"
                    name="featureCount"
                    type="number"
                    value={predictionSettings.featureCount}
                    onChange={handlePredictionSettingsChange}
                    margin="normal"
                    InputProps={{ inputProps: { min: 1, max: 100 } }}
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={predictionSettings.enableBacktesting}
                        onChange={handlePredictionSettingsChange}
                        name="enableBacktesting"
                      />
                    }
                    label="Enable Backtesting"
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={handleSavePredictionSettings}
                  >
                    Save Prediction Settings
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <NotificationsIcon sx={{ mr: 1 }} />
                  Notification Settings
                </Box>
              }
            />
            <Divider />
            <CardContent>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.enablePredictionAlerts}
                        onChange={handleNotificationSettingsChange}
                        name="enablePredictionAlerts"
                      />
                    }
                    label="Enable Prediction Alerts"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.enableOrderNotifications}
                        onChange={handleNotificationSettingsChange}
                        name="enableOrderNotifications"
                      />
                    }
                    label="Enable Order Notifications"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.enablePriceAlerts}
                        onChange={handleNotificationSettingsChange}
                        name="enablePriceAlerts"
                      />
                    }
                    label="Enable Price Alerts"
                  />
                </Grid>
                {notificationSettings.enablePriceAlerts && (
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Price Alert Threshold (%)"
                      name="priceAlertThreshold"
                      type="number"
                      value={notificationSettings.priceAlertThreshold}
                      onChange={handleNotificationSettingsChange}
                      margin="normal"
                      InputProps={{ inputProps: { min: 0.1, step: 0.1 } }}
                    />
                  </Grid>
                )}
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.enableSoundAlerts}
                        onChange={handleNotificationSettingsChange}
                        name="enableSoundAlerts"
                      />
                    }
                    label="Enable Sound Alerts"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={notificationSettings.enableBrowserNotifications}
                        onChange={handleNotificationSettingsChange}
                        name="enableBrowserNotifications"
                      />
                    }
                    label="Enable Browser Notifications"
                  />
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    startIcon={<SaveIcon />}
                    onClick={handleSaveNotificationSettings}
                  >
                    Save Notification Settings
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Account Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardHeader 
              title={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <SecurityIcon sx={{ mr: 1 }} />
                  Account & Security
                </Box>
              }
            />
            <Divider />
            <CardContent>
              <List>
                <ListItem secondaryAction={
                  <Button size="small" variant="outlined">Change</Button>
                }>
                  <ListItemIcon>
                    <SecurityIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Password" 
                    secondary="Last changed 30 days ago" 
                  />
                </ListItem>
                <ListItem secondaryAction={
                  <Button size="small" variant="outlined">Set Up</Button>
                }>
                  <ListItemIcon>
                    <DevicesIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Two-Factor Authentication" 
                    secondary="Not enabled" 
                  />
                </ListItem>
                <ListItem secondaryAction={
                  <Button size="small" variant="outlined">Manage</Button>
                }>
                  <ListItemIcon>
                    <LanguageIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Language" 
                    secondary="English (US)" 
                  />
                </ListItem>
                <ListItem secondaryAction={
                  <Button size="small" variant="outlined" color="error">
                    <DeleteIcon fontSize="small" sx={{ mr: 0.5 }} />
                    Clear
                  </Button>
                }>
                  <ListItemIcon>
                    <SettingsIcon />
                  </ListItemIcon>
                  <ListItemText 
                    primary="Clear Application Data" 
                    secondary="Reset all settings and cached data" 
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Reconnect FIX Session Dialog */}
      <Dialog
        open={reconnectDialogOpen}
        onClose={() => setReconnectDialogOpen(false)}
      >
        <DialogTitle>Reconnect FIX Session</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to reconnect the FIX session? This will temporarily disconnect from the current session and establish a new connection.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReconnectDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleReconnectFixSession} variant="contained">
            Reconnect
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default Settings;
