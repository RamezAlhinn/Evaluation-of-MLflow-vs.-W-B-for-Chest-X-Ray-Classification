"""
Simple script to start MLflow UI
Useful when mlflow CLI command is not available
"""

import subprocess
import sys
import webbrowser
import time
import os

def start_mlflow_ui(port=5000, host='127.0.0.1'):
    """Start MLflow UI server"""
    print("=" * 60)
    print("Starting MLflow UI...")
    print("=" * 60)
    print(f"Server will be available at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Start MLflow UI
        cmd = [sys.executable, "-m", "mlflow", "ui", "--port", str(port), "--host", host]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nMLflow UI server stopped.")
    except Exception as e:
        print(f"\nError starting MLflow UI: {e}")
        print("\nMake sure MLflow is installed:")
        print("  pip install mlflow")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start MLflow UI')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run MLflow UI on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run MLflow UI on (default: 127.0.0.1)')
    
    args = parser.parse_args()
    
    start_mlflow_ui(port=args.port, host=args.host)

