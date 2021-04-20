"""Test that all code blocks in README.md run without errors."""
import contextlib
from io import StringIO
from pathlib import Path
import re
import sys
from warnings import warn

import udar


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
            out = s.getvalue().strip()
            assert out == expected_out, f'{code} => {out}' 


def test_properties_documented_in_tables_actually_exist():
    failed = []
    ignore = {'Method', 'Property', '---'}
    table_blocks = re.findall(r'^### `([A-Za-z0-9]+)` object\n\n(.+?)(?=^###)',
                              README, flags=re.S | re.M)
    for obj_name, entry in table_blocks:
        obj = getattr(udar, obj_name)
        actual_attr_names = {name for name in dir(obj)
                             if not name.startswith('_')
                             and name not in ignore}
        doc_attr_names = re.findall(r'^\| (.*?) \|.*?\|.*?\|$', entry,
                                    flags=re.M)
        doc_attr_names = {name.replace(r'\_', '_') for name in doc_attr_names
                          if name not in ignore}
        bad_attr_names = doc_attr_names.difference(actual_attr_names)
        if bad_attr_names:
            failed.append(f'{obj} does not have these attrs: {bad_attr_names}')
    assert not failed


def test_all_properties_are_documented_in_tables():
    # TODO should some of the following be named with leading underscore???
    ignore_attrs = {'Document': set(),
                    'Sentence': {'analyze', 'annotation', 'depparse',
                                 'features', 'parse_cg3', 'parse_hfst',
                                 'respace', 'stress_eval', 'stress_preds2tsv',
                                 'tokenize'},
                    'Token': {'annotation', 'end_char', 'features',
                              'guess_syllable',
                              'has_tag_in_most_likely_reading', 'head',
                              'is_L2_error', 'might_be_L2_error',
                              'phon_predictions', 'phonetic_transcriptions',
                              'recase', 'start_char', 'stress_ambig',
                              'stress_eval', 'stress_predictions'},
                    'Reading': {'hfst_noL2_str'},
                    'Subreading': {'hfst_noL2_str'},
                    'Tag': {'ambig_alternative', 'is_included_in'}}
    table_blocks = re.findall(r'^### `([A-Za-z0-9]+)` object\n\n(.+?)(?=^###)',
                              README, flags=re.S | re.M)
    for obj_name, entry in table_blocks:
        ignore = ignore_attrs[obj_name]
        if ignore:
            warn(f'Ignoring the following attributes of {obj_name}: {ignore}')
        obj = getattr(udar, obj_name)
        actual_attr_names = {name for name in dir(obj)
                             if not name.startswith('_')}
        unnecessarily_ignored = ignore.difference(actual_attr_names)
        assert obj_name and len(unnecessarily_ignored) == 0, unnecessarily_ignored  # noqa: E501
        actual_attr_names = actual_attr_names - ignore
        doc_attr_names = re.findall(r'^\| (.*?) \| `.*?` \| .*?\ |$', entry,
                                    flags=re.M)
        doc_attr_names = {name.replace(r'\_', '_') for name in doc_attr_names
                          if name not in {'Method', 'Property', '---'}}
        bad_attr_names = sorted(actual_attr_name
                                for actual_attr_name in actual_attr_names
                                if actual_attr_name not in doc_attr_names)
        assert obj_name and not bad_attr_names
