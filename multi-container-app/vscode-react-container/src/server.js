const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware to serve static files from the React app
app.use(express.static(path.resolve(__dirname, 'client/build')));

// Example API route
app.get('/api/greet', (req, res) => {
    res.json({ greeting: 'Greetings from the server!' });
});

// Fallback route to serve the React app for any unmatched requests
app.get('*', (req, res) => {
    res.sendFile(path.resolve(__dirname, 'client/build', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});