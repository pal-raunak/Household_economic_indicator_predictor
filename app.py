"""
Root-level app.py for Hugging Face Spaces deployment.
This file executes the actual app from the app/ directory.
"""
import sys
import os

# Add the app directory to Python path so imports work
app_dir = os.path.join(os.path.dirname(__file__), 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Change working directory to app/ so relative paths in app/app.py work correctly
original_cwd = os.getcwd()
os.chdir(app_dir)

try:
    # Execute the app/app.py file
    with open(os.path.join(app_dir, 'app.py'), 'r', encoding='utf-8') as f:
        code = compile(f.read(), os.path.join(app_dir, 'app.py'), 'exec')
        exec(code, {'__file__': os.path.join(app_dir, 'app.py'), '__name__': '__main__'})
finally:
    # Restore original directory (though this won't matter for Streamlit)
    os.chdir(original_cwd)

