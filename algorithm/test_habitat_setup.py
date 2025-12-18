#!/usr/bin/env python3
"""
Test script to verify the habitat_main module can be imported and basic functionality works.
"""

import sys
import os

print("Testing habitat_main module...")
print("-" * 50)

# Test 1: Import check
print("\n1. Testing imports...")
try:
    from src.habitat_main import HabitatSimulator, NoMaDPolicy, waypoint_to_habitat_action

    print("   ✓ All classes imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Waypoint conversion function
print("\n2. Testing waypoint_to_habitat_action()...")
test_cases = [
    ([0.0, 0.5], "move_forward"),
    ([0.5, 0.0], "turn_right"),
    ([-0.5, 0.0], "turn_left"),
]

for waypoint, expected_action in test_cases:
    action = waypoint_to_habitat_action(waypoint)
    status = "✓" if action == expected_action else "✗"
    print(f"   {status} waypoint={waypoint} -> action={action} (expected={expected_action})")

# Test 3: Check required paths exist
print("\n3. Checking example paths...")
example_paths = {
    "Scene": "/home/liuxh/vln/nomad-habitat/tmpdata/data/versioned_data/habitat_test_scenes/skokloster-castle.glb",
    "Model Config": "/home/liuxh/vln/nomad-habitat/visualnav-transformer/train/config/nomad.yaml",
    "Model Weights": "/home/liuxh/vln/nomad-habitat/visualnav-transformer/deployment/model_weights/nomad.pth",
}

all_exist = True
for name, path in example_paths.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"   {status} {name}: {path}")
    if not exists:
        all_exist = False

if all_exist:
    print("\n   ✓ All example paths exist - you can run the full simulation!")
else:
    print("\n   ⚠ Some paths don't exist - update run_habitat_example.sh with correct paths")

# Test 4: Check dependencies
print("\n4. Checking dependencies...")
dependencies = ["habitat_sim", "habitat", "torch", "numpy", "cv2", "imageio", "yaml", "PIL"]

missing = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f"   ✓ {dep}")
    except ImportError:
        print(f"   ✗ {dep} - MISSING")
        missing.append(dep)

if missing:
    print(f"\n   ⚠ Missing dependencies: {', '.join(missing)}")
    print("   Install with: pip install <package_name>")
else:
    print("\n   ✓ All dependencies installed")

# Test 5: Check vint_train utilities
print("\n5. Checking vint_train utilities...")
try:
    from vint_train.utils import transform_images, load_model
    from vint_train.training.train_utils import get_action

    print("   ✓ vint_train utilities available")
except ImportError as e:
    print(f"   ✗ vint_train not found: {e}")
    print("   Add visualnav-transformer to PYTHONPATH:")
    print("   export PYTHONPATH=/path/to/visualnav-transformer:$PYTHONPATH")

print("\n" + "=" * 50)
print("Test completed!")
print("=" * 50)
