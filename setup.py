import setuptools
from setuptools.command.install import install
from subprocess import Popen
from sys import stderr

with open('README.md', 'r') as fh:
    long_description = fh.read()


class PostInstallCommand(install):
    """Post-installation routine for installation mode."""
    def run(self):
        install.run(self)

        try:
            Popen(['hfst-info'])
            print('HFST appears to be installed correctly.', file=stderr)
        except FileNotFoundError:
            print('Command-line hfst not found. In order to use the built-in '
                  'tokenizer, it must be installed, and in your PATH.\n'
                  'See http://divvun.no/doc/infra/compiling_HFST3.html\n',
                  file=stderr)

        try:
            import nltk
            assert nltk.download('punkt')
            print('nltk appears to be installed correctly.', file=stderr)
        except ImportError:
            print('nltk is not installed, so its tokenizer is not available.\n'
                  'Try `python3 -m pip install --user nltk`.\n', file=stderr)
        except AssertionError as e:
            print('nltk is installed, but tokenizer failed to download.', e,
                  file=stderr)
            print("Try...\n>>> import nltk\n>>> nltk.download('punkt')\n",
                  file=stderr)

        try:
            Popen(['vislcg3'])
            print('vislcg3 appears to be installed correctly.', file=stderr)
        except FileNotFoundError:
            print('vislcg3 not found. In order to perform morphosyntactic '
                  'disambiguation, it must be installed, and in your PATH.\n',
                  file=stderr)


setuptools.setup(
    name='udar',
    version='0.1.0',
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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: Flake8',
        'Framework :: Pytest',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: Russian',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Education',
        'Topic :: Education :: Computer Aided Instruction (CAI)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Utilities',
        'Typing :: Typed',
    ],
)
