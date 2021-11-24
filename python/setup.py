from setuptools import setup
from os.path import join, dirname

try:
    # obtain long description from README
    readme_path = join(dirname(__file__), "README.rst")
    with open(readme_path, encoding="utf-8") as f:
        README = f.read()
        # remove raw html not supported by PyPI
        README = "\n".join(README.split("\n")[3:])
except IOError:
    README = ""

DESCRIPTION = "Implements Demixed Principal Components Analysis"
NAME = "dPCA"
AUTHOR = "Machens Lab"
AUTHOR_EMAIL = "wieland.brendel@uni-tuebingen.de"
MAINTAINER = "Wieland Brendel"
MAINTAINER_EMAIL = "wieland.brendel@uni-tuebingen.de"
DOWNLOAD_URL = 'https://github.com/machenslab/dPCA/'
LICENSE = 'MIT'
VERSION = '1.0.5'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=README,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['dPCA'],
      package_data={},
      install_requires=['numpy', 'scipy', 'sklearn', 'numexpr', 'numba']
      )
