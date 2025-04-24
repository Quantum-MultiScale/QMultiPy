import os
import pathlib

qmultipy_data_path = os.environ.get('QMULTIPY_DATA_PATH', None)
if not qmultipy_data_path:
    qmultipy_data_path = pathlib.Path(__file__).resolve().parents[1] / 'DATA'
    if not qmultipy_data_path.is_dir():
        qmultipy_data_path = pathlib.Path(__file__).resolve().parents[2] / 'DATA'
else:
    qmultipy_data_path = pathlib.Path(qmultipy_data_path)
