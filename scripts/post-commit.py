#!/usr/bin/env python3
"""
To use this script as a local post-commit check...
    1) Symlink (or copy) this file to .git/hooks/
    2) Ensure that it is executable (chmod +x .git/hooks/post-commit).
"""
import json
import subprocess
import sys

from packaging.version import parse
import requests


sys.stdin = open('/dev/tty')

BETA = False
TEST = False

print(f'Running {__file__}...', file=sys.stderr)


def bump_python_version(pypi_version, beta=False):
    """Bump version.

    Parameters
    ----------
    pypi_version
        latest version from PyPI
    beta
        If True, output beta version
    """
    pre = ''
    base_version = list(pypi_version.release)
    if pypi_version.pre:  # if latest PyPI version is not beta
        if beta:
            pre = 'b' + str(pypi_version.pre[1] + 1)
    else:
        base_version[-1] += 1
        if beta:
            pre = 'b0'
    base_version = '.'.join(str(v) for v in base_version)
    return base_version + pre


def get_pypi_version(test=False):
    """Get version from PyPI json

    test -- if True, use test.pypi.org
    """
    if test:
        pypi_url = 'https://test.pypi.org/pypi/udar/json'
    else:
        pypi_url = 'https://pypi.org/pypi/udar/json'
    pypi_json = requests.get(pypi_url).text
    try:
        pypi_dict = json.loads(pypi_json)
        releases = [parse(version)
                    for version in pypi_dict['releases']]
        latest_release = sorted(releases)[-1]
    except:
        latest_release = parse('0.0.0')
    return latest_release


def tests():
    hv = parse('1.2.3')
    for t, expected in [('1.2.2.5', '1.2.3.0'),
                        ('1.2.2.5b4', '1.2.3.0'),
                        ('1.2.3.0', '1.2.3.1'),
                        ('1.2.3.0b4', '1.2.3.0'),
                        ('1.2.3.1', '1.2.3.2'),
                        ('1.2.3.1b4', '1.2.3.1')]:
        print('1.2.3', t, bump_python_version(hv, parse(t)), sep='\t')
        assert bump_python_version(hv, parse(t), beta=False) == expected, ('1.2.3', t, expected)
    print('BETA')
    for t, expected in [('1.2.2.5', '1.2.3.0b0'),
                        ('1.2.2.5b4', '1.2.3.0b0'),
                        ('1.2.3.0', '1.2.3.1b0'),
                        ('1.2.3.0b4', '1.2.3.0b5'),
                        ('1.2.3.1', '1.2.3.2b0'),
                        ('1.2.3.1b4', '1.2.3.1b5')]:
        print('1.2.3', t, bump_python_version(hv, parse(t), beta=True), sep='\t')
        assert bump_python_version(hv, parse(t), beta=True) == expected, ('1.2.3', t, expected)


if __name__ == '__main__':
    response = input('Do you want to tag this commit to '
                     'trigger a release on PyPI? (y/N) > ')
    if response.strip() in {'Y', 'y', 'Yes', 'YES', 'yes'}:
        pypi_version = get_pypi_version(test=TEST)
        new_version = bump_python_version(pypi_version, beta=BETA)
        print(f'Current {"Test " if TEST else ""}PyPI version:',
              pypi_version, file=sys.stderr)
        print('Suggested new version:', new_version, file=sys.stderr)
        version = input(f'Please type the version number (default: {new_version}): ')
        completed = subprocess.run(['git', 'tag', f'v{version}'])
        sys.exit(completed.returncode)
