"""Test that all code blocks in README.md run without errors."""
import contextlib
from io import StringIO
from pathlib import Path
import re
import sys


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


README = Path('README.md').read_text()
blocks = re.findall(r'```python\n(.*?)```\n', README, flags=re.M | re.S)


def test_README_import():
    assert README[:12] == '# UDAR(enie)'


def test_block_extraction():
    assert len(blocks) > 0
    assert blocks[0][:11] == 'import udar'


def test_blocks():
    for block in blocks:
        for code, expected_out in re.findall(r'((?:[^#].*\n)+)((?:# .*\n)*)',
                                             block):
            expected_out = re.sub('^# ?', '', expected_out, flags=re.M).strip()
            with stdoutIO() as s:
                exec(code, globals())
            assert s.getvalue().strip() == expected_out
