#!/bin/bash

# Kill any existing Python HTTP servers
pkill -f "python.*http.server" 2>/dev/null || true

# Get the absolute path of the project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

# Define URLs
PORT=8080
LOGIN_URL="http://localhost:$PORT/makarspace/frontend/src/pages/auth.html"
DASHBOARD_URL="http://localhost:$PORT/makarspace/frontend/src/pages/dashboard.html"

# Display info
echo "🚀 Starting MakarSpace UI Server..."
echo "📂 Project directory: $PROJECT_DIR"
echo ""
echo "🔗 Login page: $LOGIN_URL"
echo "🔗 Dashboard: $DASHBOARD_URL"
echo ""
echo "📱 UI components available:"
echo "  - Space-themed Auth page with login/register functionality"
echo "  - Interactive Dashboard with real-time anomaly detection"
echo "  - Predictive maintenance visualization"
echo "  - Explainability engine with SHAP visualizations"
echo "  - Modular architecture interface"
echo "  - Simulation tools with controls"
echo ""
echo "Press Ctrl+C to stop the server when done"
echo "---------------------------------------------------"

# Start the server and open the browser
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS
  open "$LOGIN_URL" &
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  # Linux
  xdg-open "$LOGIN_URL" &
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  # Windows
  start "$LOGIN_URL" &
fi

# Start the Python HTTP server
python3 -m http.server $PORT
