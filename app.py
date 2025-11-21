"""
Root-level app.py for Hugging Face Spaces deployment.
This file imports and runs the actual app from the app/ directory.
"""
import sys
import os

# Get paths
base_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(base_dir, 'app')
app_py = os.path.join(app_dir, 'app.py')

# Debug: Print what we're looking for
print(f"Base directory: {base_dir}")
print(f"App directory: {app_dir}")
print(f"App.py path: {app_py}")
print(f"App dir exists: {os.path.exists(app_dir)}")
print(f"App.py exists: {os.path.exists(app_py)}")
if os.path.exists(base_dir):
    print(f"Base dir contents: {os.listdir(base_dir)}")

if not os.path.exists(app_py):
    raise FileNotFoundError(
        f"app/app.py not found!\n"
        f"Base: {base_dir}\n"
        f"Looking for: {app_py}\n"
        f"Base contents: {os.listdir(base_dir) if os.path.exists(base_dir) else 'N/A'}"
    )

# Add app directory to Python path
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Change working directory to app/ for relative paths
os.chdir(app_dir)

# Import the app module
import importlib.util
spec = importlib.util.spec_from_file_location("__main__", app_py)
module = importlib.util.module_from_spec(spec)
sys.modules["__main__"] = module
spec.loader.exec_module(module)

