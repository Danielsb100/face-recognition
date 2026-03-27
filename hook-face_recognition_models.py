from PyInstaller.utils.hooks import collect_data_files

# This hook ensures that all model files from face_recognition_models are bundled
datas = collect_data_files('face_recognition_models')
