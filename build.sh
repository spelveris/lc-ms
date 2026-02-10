#!/bin/bash
# Build script for macOS/Linux

echo "Building LC-MS Analysis executable..."

# Install dependencies
pip install pyinstaller

# Build
pyinstaller \
    --name=LCMS-Analysis \
    --onedir \
    --console \
    --add-data "app:app" \
    --hidden-import=streamlit \
    --hidden-import=rainbow \
    --hidden-import=scipy \
    --hidden-import=scipy.signal \
    --hidden-import=scipy.integrate \
    --hidden-import=matplotlib \
    --hidden-import=pandas \
    --hidden-import=numpy \
    --collect-all=streamlit \
    --collect-all=altair \
    --noconfirm \
    launcher.py

echo ""
echo "Build complete! Run: ./dist/LCMS-Analysis/LCMS-Analysis"
