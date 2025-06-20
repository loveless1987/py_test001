@echo off
echo 正在安装依赖包...
pip install -r requirements.txt

echo 开始打包Windows EXE...
pyinstaller --onefile --windowed ^
    --add-data "ct_model.keras;." ^
    --add-data "sample_ct_test.mat;." ^
    --hidden-import=tensorflow ^
    --hidden-import=tensorflow.keras ^
    --hidden-import=tensorflow.keras.models ^
    --hidden-import=PyQt5 ^
    --hidden-import=PyQt5.QtCore ^
    --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=PyQt5.QtGui ^
    --hidden-import=matplotlib ^
    --hidden-import=matplotlib.backends.backend_qt5agg ^
    --hidden-import=numpy ^
    --hidden-import=scipy ^
    --hidden-import=scipy.io ^
    --hidden-import=scipy.optimize ^
    --hidden-import=pandas ^
    --name="WaveformAnalysis" ^
    "waveform_analysis(2).py"

echo 打包完成！可执行文件位于: dist\WaveformAnalysis.exe
pause 