#!/bin/bash

# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove all .coverage files
find . -type f -name ".coverage*" -exec rm -f {} +

# Remove all .pytest_cache directories
find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Remove all .egg-info directories
find . -type d -name "multidimio.egg-info" -exec rm -rf {} +

# Remove all build directories
find . -type d -name "build" -exec rm -rf {} +

# Remove all dist directories
find . -type d -name "dist" -exec rm -rf {} +

# Remove all .pytest_cache directories
find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Remove all .nox directories
find . -type d -name ".nox" -exec rm -rf {} +
