import setuptools

import setuptools_scm  # noqa: F401
import toml  # noqa: F401

setuptools.setup(package_data={'udar': ['src/udar/resources/*']},
                 setup_requires=['setuptools_scm'],
                 use_scm_version=True,
                 )
