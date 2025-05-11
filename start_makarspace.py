#!/usr/bin/env python3
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Define the server port
PORT = 8000

# Define paths to important pages
LOGIN_PATH = "makarspace/frontend/src/pages/auth.html"
DASHBOARD_PATH = "makarspace/frontend/src/pages/dashboard.html"

def start_server():
    """Start the HTTP server in a thread"""
    server_address = ('', PORT)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f"Starting server at http://localhost:{PORT}")
    print(f"Login page: http://localhost:{PORT}/{LOGIN_PATH}")
    print(f"Dashboard: http://localhost:{PORT}/{DASHBOARD_PATH}")
    httpd.serve_forever()

def open_browser():
    """Open the browser after a short delay"""
    time.sleep(1)
    login_url = f"http://localhost:{PORT}/{LOGIN_PATH}"
    print(f"Opening {login_url} in browser...")
    webbrowser.open(login_url)

if __name__ == "__main__":
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open the browser
    open_browser()
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
