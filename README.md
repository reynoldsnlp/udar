# UDAR(enie)

[![Actions Status](https://github.com/reynoldsnlp/udar/workflows/pytest/badge.svg)](https://github.com/reynoldsnlp/udar/actions)
[![codecov](https://codecov.io/gh/reynoldsnlp/udar/branch/master/graph/badge.svg)](https://codecov.io/gh/reynoldsnlp/udar)

**U**DAR **D**oes **A**ccented **R**ussian: A finite-state morphological
analyzer of Russian that handles stressed wordforms.

A python wrapper for the [Russian finite-state
transducer](https://victorio.uit.no/langtech/trunk/langs/rus/) described
originally in chapter 2 of [my dissertation](http://hdl.handle.net/10037/9685).

#### Feature requests, issues, and pull requests are welcome!

## Non-python dependencies

For all features to be available, you should have `hfst` and `vislcg3`
installed as command-line utilities. Specifically, `hfst` is needed for
FST-based tokenization, and `vislcg3` is needed for grammatical disambiguation.
The version used to successfully test the code in included in each commit in
[this file](../blob/master/hfst_vislcg3_versions.txt). The recommended method
for installing these dependencies is as follows:

#### MacOS

```bash
curl https://apertium.projectjj.com/osx/install-nightly.sh | sudo bash
```

#### Debian / Ubuntu

```bash
wget https://apertium.projectjj.com/apt/install-nightly.sh -O - | sudo bash
sudo apt-get update
sudo apt-get install cg3 hfst hfst-dev
```

## Installation

Until the first stable version is released on [PyPI](https://pypi.org/), `udar`
can be installed directly from this repository using `pip`:

```bash
$ python3 -m pip install --user git+https://github.com/reynoldsnlp/udar
```

## Introduction

The most common use-case is to use the `Document` constructor to automatically
tokenize and analyze a text. If you `print()` a `Document` object, the result
is an `XFST`/`HFST` stream:

```python
import udar
doc1 = udar.Document('Мы удивились простоте системы.')
print(doc1)
# Мы	мы+Pron+Pers+Pl1+Nom	0.000000
# 
# удивились	удивиться+V+Perf+IV+Pst+MFN+Pl	5.078125
# 
# простоте	простота+N+Fem+Inan+Sg+Dat	4.210938
# простоте	простота+N+Fem+Inan+Sg+Loc	4.210938
# 
# системы	система+N+Fem+Inan+Pl+Acc	5.429688
# системы	система+N+Fem+Inan+Pl+Nom	5.429688
# системы	система+N+Fem+Inan+Sg+Gen	5.429688
# 
# .	.+CLB	0.000000
```

## Data objects

### `Document` object

| Property | Type | Description |
| --- | --- | --- |
| text | `str` | Original text of this document |
| sentences | `List[Sentence]` | List of sentences in this document |
| num\_tokens | `int` | Number of tokens in this document |
| features | `tuple` | `udar.features.FeatureExtractor` stores extracted features here |

`Document` objects have convenient methods for adding stress or converting to
phonetic transcription.

| Method | Return type | Description |
| --- | --- | --- |
| stressed | `str` | The original text of the document with stress marks |
| phonetic | `str` | The original text converted to phonetic transcription |
| transliterate | `str` | The original text converted to Romanized Cyrillic (default=Scholarly) |
| disambiguate | `None` | Disambiguate readings using the Constraint Grammar |
| cg3\_str | `str` | Analysis stream in the [VISL-CG3 format](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| from\_cg3 | `Document` | Create `Document` from [VISL-CG3 format stream](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| hfst\_str | `str` | Analysis stream in the XFST/HFST format |
| from\_hfst | `Document` | Create `Document` from XFST/HFST format stream |

#### Examples

```python
stressed_doc1 = doc1.stressed()
print(stressed_doc1)
# Мы́ удиви́лись простоте́ систе́мы.

phonetic_doc1 = doc1.phonetic()
print(phonetic_doc1)
# мы́ уд'ив'и́л'ис' пръстʌт'э́ с'ис'т'э́мы.
```

### `Sentence` object

| Property | Type | Description |
| --- | --- | --- |
| doc | `Document` | "Back pointer" to the parent document of this sentence |
| text | `str` | Original text of this sentence |
| tokens | `List[Token]` | The list of tokens in this sentence |
| id | `str` | (optional) Sentence id, if assigned at creation |

| Method | Return type | Description |
| --- | --- | --- |
| stressed | `str` | The original text of the sentence with stress marks |
| phonetic | `str` | The original text converted to phonetic transcription |
| transliterate | `str` | The original text converted to Romanized Cyrillic (default=Scholarly) |
| disambiguate | `None` | Disambiguate readings using the Constraint Grammar |
| cg3\_str | `str` | Analysis stream in the [VISL-CG3 format](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| from\_cg3 | `Sentence` | Create `Sentence` from [VISL-CG3 format stream](https://visl.sdu.dk/cg3/single/#stream-vislcg)
| hfst\_str | `str` | Analysis stream in the XFST/HFST format |
| from\_hfst | `Sentence` | Create `Sentence` from XFST/HFST format stream |

### `Token` object

| Property | Type | Description |
| --- | --- | --- |
| id | `str` | The index of this token in the sentence, 1-based |
| text | `str` | The original text of this token |
| misc | `str` | Miscellaneous annotations with regard to this token |
| lemmas | `Set[str]` | All possible lemmas, based on remaining readings |
| most\_likely\_reading | `Reading` | "Most likely" reading |
| most\_likely\_lemma | `str` | Lemma from the "most likely" reading |
| readings | `List[Reading]` | List of readings not removed by the Constraint Grammar |
| removed\_readings | `List[Reading]` | List of readings removed by the Constraint Grammar | head | `int` | The id of the syntactic head of this token in the sentence, 1-based (0 is reserved for an artificial symbol that represents the root of the syntactic tree). |
| deprel | `str` | The dependency relation between this word and its syntactic head. Example: ‘nmod’. |

| Method | Return type | Description |
| --- | --- | --- |
| stresses | `Set[str]` | All possible stressed wordforms, based on remaining readings |
| stressed | `str` | The original text of the sentence with stress marks |
| phonetic | `str` | The original text converted to phonetic transcription |
| transliterate | `str` | The original text converted to Romanized Cyrillic (default=Scholarly) |
| cg3\_str | `str` | Analysis stream in the [VISL-CG3 format](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| hfst\_str | `str` | Analysis stream in the XFST/HFST format |

### `Reading` object

| Property | Type | Description |
| --- | --- | --- |
| lemma | `str` | The lemma of the reading |
| tags | `List[Tag]` | The part-of-speech, morphosyntactic, semantic and other tags |
| tagset | `Set[Tag]` | Same as tags, but for faster membership testing (`in` Reading) |
| weight | `str` | Weight indicating the likelihood of the reading, without respect to context |
| cg\_rule | `str` | Reference to the rule in the constraint grammar that removed/selected/etc. this reading |
| most\_likely | `bool` | Indicates whether this reading has been selected as the most likely |

| Method | Return type | Description |
| --- | --- | --- |
| cg3\_str | `str` | Analysis stream in the [VISL-CG3 format](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| hfst\_str | `str` | Analysis stream in the XFST/HFST format |
| generate | `str` | Generate the wordform from this reading |
| replace\_tag | `None` | Replace a tag in this reading |

### `Tag` object

| Property | Type | Description |
| --- | --- | --- |
| name | `str` | The name of this tag |
| ms\_feat | `str` | Morphosyntactic feature that this tag is associated with (e.g. `Dat` has ms\_feat `CASE`) |
| detail | `str` | Description of the tag's purpose or meaning |
| is\_L2 | `bool` | Whether this tag indicates a second-language learner error |

| Method | Return type | Description |
| --- | --- | --- |
| info | `str` | Alias for `Tag.detail` |


### Convenience functions

A number of functions are included, both for convenience, and to give concrete
examples for using the API.

#### noun\_distractors()

This function generates all six cases of a given noun. If the given noun is
singular, then the function generates singular forms. If the given noun is
plural, then the function generates plural forms. Such a list can be used in a
multiple-choice exercise, hence the name `distractors`.

```python
sg_paradigm = udar.noun_distractors('словом')
print(sg_paradigm == {'сло́ву', 'сло́ве', 'сло́вом', 'сло́ва', 'сло́во'})
# True

pl_paradigm = udar.noun_distractors('словах')
print(pl_paradigm == {'слова́м', 'слова́', 'слова́х', 'слова́ми', 'сло́в'})
# True
```

If unstressed forms are desired, simply pass the argument `stressed=False`.

#### diagnose\_L2()

This function will take a text string as the argument, and will return a
dictionary of all the types of L2 errors in the text, along with examples of
the error.

```python
diag = udar.diagnose_L2('Мы разговаривали в кафетерие с Таной')
print(diag == {'Err/L2_ii': {'кафетерие'}, 'Err/L2_Pal': {'Таной'}})
# True
```

#### tag\_info()

This function will look up the meaning of any tag used by the analyzer.

```python
print(udar.tag_info('Err/L2_ii'))
# L2 error: Failure to change ending ие to ии in +Sg+Loc or +Sg+Dat, e.g. к Марие, о кафетерие, о знание
```

### Using the transducer manually

The transducer itself is the `Udar` class, which can be initialized as one of
four flavors:

1. `L2-analyzer` [default]: General analyzer with second-language learner
   errors added
1. `analyzer`: General-purpose analyzer
1. `generator`: Generator of unstressed wordforms
1. `accented-generator`: Generator of stressed wordforms

```python
analyzer = udar.Udar('analyzer')
```

The `Udar.lookup()` method takes a token `str` and returns a `Token`.

```python
token1 = analyzer.lookup('сло́ва')
print(token1)
# сло́ва [слово_N_Neu_Inan_Sg_Gen]

token2 = analyzer.lookup('слова')
print(token2)
# слова [слово_N_Neu_Inan_Pl_Acc  слово_N_Neu_Inan_Pl_Nom  слово_N_Neu_Inan_Sg_Gen]
```

## Working with `Token`s and `Readings`s

You can easily check if a lemma or morphosyntactic tag are in a `Token` or
`Reading` using `in`:
 
```python
print(token2)
# слова [слово_N_Neu_Inan_Pl_Acc  слово_N_Neu_Inan_Pl_Nom  слово_N_Neu_Inan_Sg_Gen]

print('Gen' in token2)  # do any of the readings include Genitive case?
# True

print('слово' in token2)  # do any of the readings have the lemma 'слово'?
# True

print('новый' in token2)
# False
```

You can make a filtered list of a `Token`'s readings using the following idiom:

```python
pl_readings = [reading for reading in token2 if 'Pl' in reading]
print(pl_readings)
# [Reading(слово+N+Neu+Inan+Pl+Acc, 5.975586, ), Reading(слово+N+Neu+Inan+Pl+Nom, 5.975586, )]
```

## Related projects

https://github.com/mikahama/uralicNLP
