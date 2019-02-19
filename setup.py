import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='udar',
    version='0.0.1',
    author='Robert Reynolds',
    author_email='ReynoldsRJR@gmail.com',
    description='Detailed part-of-speech tagger for (accented) Russian.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/reynoldsnlp/udar',
    packages=setuptools.find_packages(),
    dependency_links=['https://github.com/ljos/pyvislcg3'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
)
