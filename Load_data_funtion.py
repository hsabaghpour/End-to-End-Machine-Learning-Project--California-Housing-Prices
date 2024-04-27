"""
Summary: The function load_housing_data() searches for the file housing.tgz. 
If it's absent, it generates a directory named datasets within the current directory (/content by default in Colab). 
Then, it downloads housing.tgz from ageron/data GitHub repository, unzips its contents into datasets, 
forming the datasets/housing directory with housing.csv inside. Finally, 
the function loads this CSV into a Pandas DataFrame and returns it.

"""

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
