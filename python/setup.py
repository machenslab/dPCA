from distutils.core import setup
from Cython.Build import cythonize
import numpy

DESCRIPTION = "Implements Demixed Principal Components Analysis"
LONG_DESCRIPTION = DESCRIPTION
NAME = "dPCA"
AUTHOR = "Machens Lab"
AUTHOR_EMAIL = "wieland.brendel@neuro.fchampalimaud.org"
MAINTAINER = "Wieland Brendel"
MAINTAINER_EMAIL = "wieland.brendel@neuro.fchampalimaud.org"
DOWNLOAD_URL = 'https://github.com/machenslab/dPCA/'
LICENSE = 'MIT'
VERSION = '0.1'

ext_module = cythonize("dPCA/nan_shuffle.pyx")
ext_module[0].include_dirs = [numpy.get_include(), '.']

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['dPCA'],
      package_data={},
      requires=['sklearn', 'numexpr'],
      ext_modules=ext_module
      )
