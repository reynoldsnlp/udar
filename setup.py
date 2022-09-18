import site
import sys

import setuptools


# workaround for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = '--user' in sys.argv[1:]

setuptools.setup(package_data={'udar': ['src/udar/resources/*']},
                 setup_requires=['setuptools_scm'],
                 use_scm_version=True,
                 )
