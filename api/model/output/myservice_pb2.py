import sys
import os

# Get absolute path of the output directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

print("BASE_DIR:", BASE_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# Add to sys.path if not already present
if OUTPUT_DIR not in sys.path:
    sys.path.insert(0, OUTPUT_DIR)

print("Updated sys.path:", sys.path)

# Try importing
try:
    import api.model.output.myservice_pb2 as myservice_pb2
    print("✅ Import successful!")
except ModuleNotFoundError as e:
    print(f"❌ Error importing myservice_pb2: {e}")
    print("Ensure 'output/' contains 'myservice_pb2.py' and is in the Python path.")
