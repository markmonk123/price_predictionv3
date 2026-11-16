# Price Prediction Platform - Diagnostics Report
**Generated**: 2026-11-09

## Environment Status
✅ Node.js: v24.22.1
✅ npm: 11.19.1
✅ Python: 3.15.0
✅ Dependencies: Installed (npm + Python)

## Project Structure
✅ Backend:
  - server.js (main entry)
  - server/ (Express app, routes, services)
  - services/predictionService.js (Python model runner)
  - python/run_prediction.py (Model executor)
  
✅ Frontend:
  - client/ (React app with Vite)
  - client/src/components/PredictionCard.jsx (Run button component)
  
✅ MCP Server:
  - mcp/server.js (Model Context Protocol server)
  - Configured in .mcp.json
  - Provides run_prediction tool
  
## API Endpoints
✅ GET  /api/v1/health          → Server status, model running state
✅ GET  /api/v1/predictions/latest → Get last prediction
✅ POST /api/v1/predictions/run → Run model now (requires auth if API_TOKEN set)

## Known Issues & Solutions

### Issue: "Run model" button is inactive
**Possible Causes:**
1. Server not running on port 5000
2. Model is stuck in "running" state from health check
3. Python dependencies not fully installed
4. run_prediction.py has import errors

**Diagnostic Steps:**
```bash
# 1. Check if backend server is running
curl http://localhost:5000/api/v1/health

# 2. If server not responding, start it:
npm start          # Just backend
npm run dev        # Backend with auto-reload
npm run dev-full   # Backend + client together

# 3. Check if client can reach API (browser console):
fetch('http://localhost:5000/api/v1/health').then(r => r.json()).then(console.log)

# 4. Test run endpoint directly:
curl -X POST http://localhost:5000/api/v1/predictions/run

# 5. Check Python dependencies:
python -c "import sklearn, pandas, numpy; print('OK')"

# 6. Test run_prediction.py directly:
cd python && python run_prediction.py --help
```

## Ports
- Backend: http://localhost:5000 (API at /api/v1)
- Frontend: http://localhost:5173
- MCP Server: Runs on stdio (for VS Code/Claude Code)

## To Resume Development
```bash
npm run dev-full    # Start both servers
# Open http://localhost:5173 in browser
```

