# Habitat Simulation with NoMaD Navigation

This directory contains a complete implementation for running visual navigation in Habitat-Sim using the NoMaD (Nomadic) diffusion-based policy.

## Overview

The implementation consists of:

1. **`habitat_main.py`** - Main script that runs the simulation
2. **`run_habitat_example.sh`** - Example shell script to run the simulation
3. **`environment.ipynb`** - Original Jupyter notebook with code exploration

## Architecture

### Classes

#### `HabitatSimulator`
Wrapper class for Habitat-Sim that handles:
- Simulator configuration and initialization
- Agent setup and positioning
- Observation retrieval (RGB images)
- Top-down map generation
- Trajectory visualization
- World-to-map coordinate conversion

#### `NoMaDPolicy`
Visual navigation policy using the NoMaD diffusion model:
- Loads pre-trained model weights
- Maintains context queue of recent observations
- Predicts waypoints using diffusion-based denoising
- Handles exploration mode (no explicit goal)

### Key Functions

- **`waypoint_to_habitat_action()`** - Converts predicted waypoints to discrete Habitat actions
- **`main()`** - Main simulation loop that integrates simulator and policy

## Requirements

```bash
# Core dependencies
habitat-sim
habitat-lab
torch
numpy
opencv-python
imageio
pyyaml
diffusers
pillow

# vint_train utilities (from visualnav-transformer repo)
# Make sure to add the path to your Python path
```

## Usage

### Basic Usage

```bash
python habitat_main.py \
    --scene-path /path/to/scene.glb \
    --model-config /path/to/nomad.yaml \
    --model-weights /path/to/nomad.pth \
    --num-steps 1000 \
    --save-video \
    --output-video exploration.mp4
```

### Using the Example Script

1. Edit `run_habitat_example.sh` with your paths:
   ```bash
   SCENE_PATH="your/scene/path.glb"
   MODEL_CONFIG="your/config/nomad.yaml"
   MODEL_WEIGHTS="your/weights/nomad.pth"
   ```

2. Make it executable and run:
   ```bash
   chmod +x run_habitat_example.sh
   ./run_habitat_example.sh
   ```

### Command-Line Arguments

#### Required Arguments

- `--scene-path`: Path to Habitat scene file (`.glb` format)
- `--model-config`: Path to model configuration YAML file
- `--model-weights`: Path to model checkpoint (`.pth` file)

#### Navigation Parameters

- `--num-steps`: Number of simulation steps (default: 1000)
- `--waypoint-index` / `-w`: Index of waypoint for navigation, 0-4 (default: 2)
- `--num-samples` / `-n`: Number of action samples from diffusion model (default: 8)
- `--context-warmup`: Steps of random exploration before using policy (default: 10)

#### Device Configuration

- `--device`: Device to run model on - `cuda` or `cpu` (default: cuda)

#### Video Output

- `--save-video`: Flag to enable video saving
- `--output-video`: Output video filename (default: habitat_exploration.mp4)
- `--video-fps`: Video framerate (default: 10)

## How It Works

### 1. Initialization Phase
- Loads Habitat scene
- Initializes agent at random navigable position
- Generates static top-down map for visualization
- Loads NoMaD model and diffusion scheduler

### 2. Exploration Loop
For each step:
1. **Observe**: Get RGB image from agent's camera
2. **Context Building**: Add observation to context queue
3. **Warmup Phase** (first N steps): Random actions for context initialization
4. **Policy Phase**:
   - Transform context images
   - Run diffusion denoising to predict trajectory waypoints
   - Select waypoint and convert to Habitat action
5. **Execute**: Apply action to simulator
6. **Visualize**: Update trajectory on top-down map
7. **Record**: Save combined RGB + map frame for video

### 3. Output
- Saves video showing side-by-side:
  - Left: First-person RGB view
  - Right: Top-down map with trajectory in red

## Waypoint to Action Conversion

The `waypoint_to_habitat_action()` function converts continuous waypoint predictions to discrete Habitat actions:

- Predicted waypoints are `[delta_x, delta_y]` in normalized coordinates
- **Forward/Backward**: If `|delta_y|` is dominant and positive → `move_forward`
- **Turn Left**: If `delta_x` is dominant and negative → `turn_left`
- **Turn Right**: If `delta_x` is dominant and positive → `turn_right`

You can customize these thresholds and logic based on your robot's motion model.

## Differences from ROS Version

The original `environment.ipynb` contains a ROS-based implementation. This standalone version:

✅ **Removed**: ROS dependencies (rospy, ROS messages)
✅ **Added**: Direct Habitat-Sim integration
✅ **Added**: Standalone observation handling (no ROS topics)
✅ **Added**: Action conversion logic for Habitat
✅ **Kept**: Same NoMaD diffusion policy logic
✅ **Kept**: Visualization and trajectory recording

## Example Output

After running, you'll get:
- `habitat_exploration.mp4`: Video showing the exploration
  - Left panel: Agent's first-person view
  - Right panel: Top-down map with red trajectory line

## Troubleshooting

### "Model weights not found"
Make sure the paths in your command or script point to the correct files.

### "Scene file not found"
Download Habitat test scenes from: https://github.com/facebookresearch/habitat-sim#datasets

### CUDA out of memory
Try reducing `--num-samples` or use `--device cpu`

### Import errors for vint_train
Make sure the visualnav-transformer repository is in your Python path:
```bash
export PYTHONPATH=/path/to/visualnav-transformer:$PYTHONPATH
```

## Customization

### Change Navigation Behavior

Edit `waypoint_to_habitat_action()` to modify how waypoints map to actions.

### Use Different Models

Replace model paths to use `vint`, `gnm`, or `late_fusion` instead of `nomad`.

### Add More Sensors

Modify `HabitatSimulator.make_sim_config()` to add depth, semantic, or other sensors.

### Custom Heuristics

In `NoMaDPolicy.predict_waypoint()`, change how action samples are selected (currently uses first sample).

## References

- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- [NoMaD Paper](https://general-navigation-models.github.io/nomad/)
- [ViNT (Visual Navigation Transformer)](https://github.com/robodhruv/visualnav-transformer)

## License

Same as the parent project.
