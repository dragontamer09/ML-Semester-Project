# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('load_data.py', '.'), ('preprocess.py', '.'), ('feature_metadata.py', '.'), ('evaluate.py', '.'), ('C:\\Users\\Dragon\\Documents\\Repositories\\ML-Semester-Project\\.venv\\Lib\\site-packages\\imblearn\\VERSION.txt', 'imblearn')]
binaries = []
hiddenimports = ['sklearn.impute', 'sklearn.tree', 'sklearn.svm', 'sklearn.calibration', 'sklearn.preprocessing', 'sklearn.pipeline', 'sklearn.compose', 'sklearn.metrics', 'sklearn.model_selection']
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['train.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='train',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
