@echo off
REM Windows build script for LC-MS Analysis

echo Building LC-MS Analysis for Windows...

pip install pyinstaller

pyinstaller ^
    --name=LCMS-Analysis ^
    --onedir ^
    --console ^
    --add-data "app;app" ^
    --copy-metadata streamlit ^
    --copy-metadata altair ^
    --collect-data streamlit ^
    --hidden-import=streamlit ^
    --hidden-import=streamlit.runtime.scriptrunner ^
    --hidden-import=streamlit.runtime.scriptrunner.magic_funcs ^
    --hidden-import=streamlit.web.cli ^
    --hidden-import=rainbow ^
    --hidden-import=scipy.signal ^
    --hidden-import=scipy.integrate ^
    --hidden-import=matplotlib.backends.backend_svg ^
    --hidden-import=matplotlib.backends.backend_pdf ^
    --hidden-import=matplotlib.backends.backend_agg ^
    --exclude-module=torch ^
    --exclude-module=tensorflow ^
    --exclude-module=keras ^
    --exclude-module=IPython ^
    --exclude-module=jupyter ^
    --exclude-module=notebook ^
    --exclude-module=pytest ^
    --exclude-module=sphinx ^
    --exclude-module=docutils ^
    --exclude-module=pygments ^
    --exclude-module=jedi ^
    --exclude-module=parso ^
    --exclude-module=black ^
    --exclude-module=mypy ^
    --exclude-module=pylint ^
    --exclude-module=cv2 ^
    --exclude-module=opencv ^
    --exclude-module=PIL.ImageQt ^
    --exclude-module=PyQt5 ^
    --exclude-module=PyQt6 ^
    --exclude-module=PySide2 ^
    --exclude-module=PySide6 ^
    --exclude-module=tkinter ^
    --exclude-module=_tkinter ^
    --noconfirm ^
    launcher.py

echo.
echo Build complete!
echo Output: dist\LCMS-Analysis\
