#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "------------------------------------------------"
echo "Setup complete!"
echo ""
echo "NOTE: You need to manually place yolo12l-person-seg.pt in this directory."
echo ""
echo "To run the pipeline:"
echo "  source venv/bin/activate"
echo "  python3 ratman_pipeline.py --input_video <path> --reference_image <path> --output_video output.mp4"
echo "------------------------------------------------"
