#!/bin/bash

set -e  # Exit on any error

echo "=== Zarr Hexdump Bootstrap Script ==="
echo

# Check if Rust is installed, if not install it
if ! command -v rustc &> /dev/null; then
    echo "Rust is not installed. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo "✓ Rust installed successfully"
fi

echo "✓ Rust is installed ($(rustc --version))"

# Check if Cargo is available
if ! command -v cargo &> /dev/null; then
    echo "Sourcing Rust environment..."
    source $HOME/.cargo/env
fi

echo "✓ Cargo is available ($(cargo --version))"
echo

# Install blosc library if not present
echo "Checking for blosc library..."
if ! pkg-config --exists blosc; then
    echo "Installing blosc library..."
    sudo apt-get update
    sudo apt-get install -y libblosc-dev pkg-config
    echo "✓ Blosc library installed"
else
    echo "✓ Blosc library already available"
fi
echo

# Build the project
echo "Building the Zarr hexdump tool..."
if cargo build --release; then
    echo "✓ Build successful!"
else
    echo "✗ Build failed!"
    exit 1
fi

echo
echo "=== Usage ==="
echo "To hexdump a Zarr array, run:"
echo "  ./target/release/zarr-hexdump /path/to/your/zarr/array"
echo
echo "Or use cargo run:"
echo "  cargo run --release -- /path/to/your/zarr/array"
echo

# Check if a path was provided as an argument
if [ $# -eq 1 ]; then
    ZARR_PATH="$1"
    echo "=== Running hexdump on provided path: $ZARR_PATH ==="
    echo

    if [ -e "$ZARR_PATH" ]; then
        ./target/release/zarr-hexdump "$ZARR_PATH"
    else
        echo "Error: Path '$ZARR_PATH' does not exist!"
        exit 1
    fi
elif [ $# -gt 1 ]; then
    echo "Error: Too many arguments provided."
    echo "Usage: $0 [zarr_array_path]"
    exit 1
else
    echo "No Zarr array path provided. Build completed successfully."
    echo "Use the commands above to run the hexdump tool."
fi
