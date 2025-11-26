#!/bin/bash

# Example script to run Habitat simulation with NoMaD policy
# Modify these paths according to your setup

SCENE_PATH="/home/liuxh/vln/vint_docker_env/tmpdata/data/versioned_data/habitat_test_scenes/skokloster-castle.glb"
MODEL_CONFIG="/home/liuxh/vln/vint_docker_env/visualnav-transformer/train/config/nomad.yaml"
MODEL_WEIGHTS="/home/liuxh/vln/vint_docker_env/visualnav-transformer/deployment/model_weights/nomad.pth"

# Run the simulation
python habitat_main.py \
    --scene-path "$SCENE_PATH" \
    --model-config "$MODEL_CONFIG" \
    --model-weights "$MODEL_WEIGHTS" \
    --num-steps 600 \
    --waypoint-index 2 \
    --num-samples 8 \
    --context-warmup 10 \
    --device cuda \
    --save-video \
    --output-video habitat_exploration.mp4 \
    --video-fps 30

echo "Simulation completed! Check habitat_exploration.mp4 for results."
