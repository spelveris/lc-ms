"""Build script to create executable using PyInstaller."""

import subprocess
import sys
import os


def main():
    print("Building LC-MS Analysis executable...")
    print("=" * 50)

    # Install PyInstaller if not present
    print("Checking PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=LCMS-Analysis",
        "--onedir",  # Use onedir for faster startup (onefile is slower)
        "--console",  # Keep console to see output/errors
        "--add-data=app:app",  # Include app folder
        "--hidden-import=streamlit",
        "--hidden-import=rainbow",
        "--hidden-import=scipy",
        "--hidden-import=scipy.signal",
        "--hidden-import=scipy.integrate",
        "--hidden-import=matplotlib",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=altair",
        "--hidden-import=PIL",
        "--collect-all=streamlit",
        "--collect-all=altair",
        "--noconfirm",
        "launcher.py"
    ]

    print("Running PyInstaller...")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print()
        print("=" * 50)
        print("Build complete!")
        print("Executable is in: dist/LCMS-Analysis/")
        print("Run: dist/LCMS-Analysis/LCMS-Analysis.exe (Windows)")
        print("     dist/LCMS-Analysis/LCMS-Analysis (macOS/Linux)")
    else:
        print("Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
