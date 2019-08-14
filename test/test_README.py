"""Test that all code blocks in README.md run without errors."""
from pathlib import Path
import re


README = Path('README.md').read_text()
blocks = re.findall(r'```python\n(.*?)```\n', README, flags=re.M | re.S)


def test_README_import():
    assert README[:12] == '# UDAR(enie)'


def test_block_extraction():
    assert len(blocks) > 0
    assert blocks[0][:11] == 'import udar'


def test_blocks():
    """Just checks that nothing throws errors."""
    for block in blocks:
        exec(block, globals())
