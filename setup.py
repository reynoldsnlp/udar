import setuptools
from setuptools.command.install import install
from subprocess import Popen

with open('README.md', 'r') as fh:
    long_description = fh.read()


class PostInstallCommand(install):
    """Post-installation routine for installation mode."""
    def run(self):
        install.run(self)
        try:
            Popen(['hfst-info'])
        except FileNotFoundError:
            print('Command-line hfst not found. In order to use the built-in '
                  'tokenizer, it must be installed, and in your PATH.')
            print('See http://divvun.no/doc/infra/compiling_HFST3.html\n')

        try:
            import nltk
            assert nltk.download('punkt')
        except ImportError:
            print('nltk is not installed, so its tokenizer is not available.')
            print('Try `python3 -m pip install --user nltk`.\n')
        except AssertionError as e:
            print('nltk is installed, but tokenizer failed to download.', e)
            print("Try...\n>>> import nltk\n>>>nltk.download('punkt')\n")

        try:
            Popen(['vislcg3'])
        except FileNotFoundError:
            print('vislcg3 not found. In order to perform morphosyntactic '
                  'disambiguation, it must be installed, and in your PATH.\n')


setuptools.setup(
    name='udar',
    version='0.0.1',
    author='Robert Reynolds',
    author_email='ReynoldsRJR@gmail.com',
    cmdclass={'install': PostInstallCommand},
    description='Detailed part-of-speech tagger for (accented) Russian.',
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/reynoldsnlp/udar',
    packages=setuptools.find_packages(),
    package_dir={'udar': 'udar'},
    package_data={'udar': ['udar/resources/*']},
    install_requires=['hfst', 'nltk'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    # dependency_links=['https://github.com/ljos/pyvislcg3'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
)
