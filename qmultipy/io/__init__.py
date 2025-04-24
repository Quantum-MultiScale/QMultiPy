from pathlib import Path

from .io import (
    read,
    read_all,
    read_data,
    read_density,
    read_potential,
    write,
    write_all,
    write_density,
    write_potential,
)


def download_files(files):
    import urllib.request

    if isinstance(files, (str, Path)):
        files = {Path(files).name: files}
    elif isinstance(files, (list, tuple)):
        files = {Path(file).name: file for file in files}
    for file, url in files.items():
        if not Path(file).is_file():
            urllib.request.urlretrieve(url, file)
