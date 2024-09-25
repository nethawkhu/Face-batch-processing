from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# 收集所有的 submodules
hiddenimports = collect_submodules('mediapipe')

# 收集所有的数据文件
datas = collect_data_files('mediapipe')