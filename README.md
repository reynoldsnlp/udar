# UDAR(enie)

[![Actions Status](https://github.com/reynoldsnlp/udar/workflows/Test%20and%20Publish/badge.svg)](https://github.com/reynoldsnlp/udar/actions)
[![codecov](https://codecov.io/gh/reynoldsnlp/udar/branch/master/graph/badge.svg)](https://codecov.io/gh/reynoldsnlp/udar)

**U**DAR **D**oes **A**ccented **R**ussian: A finite-state morphological
analyzer of Russian that handles stressed wordforms.

A python wrapper for the [Russian finite-state
transducer](https://victorio.uit.no/langtech/trunk/langs/rus/) described
originally in chapter 2 of [my dissertation](http://hdl.handle.net/10037/9685).

If you use this work in your research please cite the following:

---
> Reynolds, Robert J. "Russian natural language processing for computer-assisted language learning: capturing the benefits of deep morphological analysis in real-life applications" PhD Diss., UiT–The Arctic University of Norway, 2016. https://hdl.handle.net/10037/9685
---

#### Feature requests, issues, and pull requests are welcome!

## Non-python dependencies

This repository uses `git-lfs` for large files. Be sure to
[download and install it](https://git-lfs.github.com/) before you clone/commit.

For all features to be available, you should have `hfst` and `vislcg3`
installed as command-line utilities. Specifically, `hfst` is needed for
FST-based tokenization, and `vislcg3` is needed for grammatical disambiguation.
The version used to successfully test the code is included in each commit in
[this file](../master/hfst_vislcg3_versions.txt). The recommended method
for installing these dependencies is as follows:

#### MacOS

```bash
$ curl https://apertium.projectjj.com/osx/install-nightly.sh | sudo bash
```

#### Debian / Ubuntu

```bash
$ wget https://apertium.projectjj.com/apt/install-nightly.sh -O - | sudo bash
$ sudo apt-get install cg3 hfst hfst-dev
```

## Installation

Until the first stable version is released on [PyPI](https://pypi.org/), `udar`
can be installed directly from this repository using `pip`:

```bash
$ python3 -m pip install --user git+https://github.com/reynoldsnlp/udar
```

## Introduction

> NB! Documentation is currently limited to docstrings. I recommend that you
> use `help()` frequently to see how to use classes and methods. For example,
> to see what options are available for building a `Document`, try
> `help(Document)`.

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

Passing the argument `disambiguate=True`, or running `doc1.disambiguate()`
after the fact will run a Constraint Grammar to remove as many ambiguous
readings as possible. ***This grammar is far from complete, so some ambiguous
readings will remain.***

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
| to\_dict | `list` | Convert to a complex list object |
| to\_json | `str` | Convert to a JSON object |

#### Examples

```python
stressed_doc1 = doc1.stressed()
print(stressed_doc1)
# Мы́ удиви́лись простоте́ систе́мы.

ambig_doc = udar.Document('Твои слова ничего не значат.', disambiguate=True)
print(sorted(ambig_doc[1].stresses()))  # Note that слова is still ambiguous
# ['сло́ва', 'слова́']

print(ambig_doc.stressed(selection='safe'))  # 'safe' skips сло́ва and слова́
# Твои́ слова ничего́ не зна́чат.
print(ambig_doc.stressed(selection='all'))  # 'all' combines сло́ва and слова́
# Твои́ сло́ва́ ничего́ не зна́чат.
print(ambig_doc.stressed(selection='rand') in {'Твои́ сло́ва ничего́ не зна́чат.', 'Твои́ слова́ ничего́ не зна́чат.'})  # 'rand' randomly chooses between сло́ва and слова́
# True


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
| to\_dict | `list` | Convert to a complex list object |

### `Token` object

| Property | Type | Description |
| --- | --- | --- |
| id | `str` | The index of this token in the sentence, 1-based |
| text | `str` | The original text of this token |
| misc | `str` | Miscellaneous annotations with regard to this token |
| lemmas | `Set[str]` | All possible lemmas, based on remaining readings |
| readings | `List[Reading]` | List of readings not removed by the Constraint Grammar |
| removed\_readings | `List[Reading]` | List of readings removed by the Constraint Grammar | head | `int` | The id of the syntactic head of this token in the sentence, 1-based (0 is reserved for an artificial symbol that represents the root of the syntactic tree). |
| deprel | `str` | The dependency relation between this word and its syntactic head. Example: ‘nmod’. |

| Method | Return type | Description |
| --- | --- | --- |
| stresses | `Set[str]` | All possible stressed wordforms, based on remaining readings |
| stressed | `str` | The original text of the sentence with stress marks |
| phonetic | `str` | The original text converted to phonetic transcription |
| most\_likely\_reading | `Reading` | "Most likely" reading (may be partially random selection) |
| most\_likely\_lemmas | `List[str]` | List of lemma(s) from the "most likely" reading |
| transliterate | `str` | The original text converted to Romanized Cyrillic (default=Scholarly) |
| force\_disambiguate | `None` | Fully disambiguate readings using methods **other than** the Constraint Grammar |
| cg3\_str | `str` | Analysis stream in the [VISL-CG3 format](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| hfst\_str | `str` | Analysis stream in the XFST/HFST format |
| to\_dict | `dict` | Convert to a `dict` object |

### `Reading` object

| Property | Type | Description |
| --- | --- | --- |
| subreadings | `List[Subreading]` | Usually only one subreading, but multiple subreadings are possible for complex `Token`s. |
| lemmas | `List[str]` | Lemmas from all subreadings |
| grouped\_tags | `List[Tag]` | The part-of-speech, morphosyntactic, semantic and other tags from all subreadings |
| weight | `str` | Weight indicating the likelihood of the reading, without respect to context |
| cg\_rule | `str` | Reference to the rule in the constraint grammar that removed/selected/etc. this reading. If no action has been taken on this reading, then `''`. |
| is\_most\_likely | `bool` | Indicates whether this reading has been selected as the most likely reading of its `Token`. Note that some selection methods may be at least partially ***random***. |

| Method | Return type | Description |
| --- | --- | --- |
| cg3\_str | `str` | Analysis stream in the [VISL-CG3 format](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| hfst\_str | `str` | Analysis stream in the XFST/HFST format |
| generate | `str` | Generate the wordform from this reading |
| replace\_tag | `None` | Replace a tag in this reading |
| does\_not\_conflict | `bool` | Determine whether reading from external tagset (e.g. Universal Dependencies) conflicts with this reading |
| to\_dict | `list` | Convert to a `list` object |

### `Subreading` object

| Property | Type | Description |
| --- | --- | --- |
| lemma | `str` | The lemma of the subreading |
| tags | `List[Tag]` | The part-of-speech, morphosyntactic, semantic and other tags |
| tagset | `Set[Tag]` | Same as `tags`, but for faster membership testing (`in` Reading) |

| Method | Return type | Description |
| --- | --- | --- |
| cg3\_str | `str` | Analysis stream in the [VISL-CG3 format](https://visl.sdu.dk/cg3/single/#stream-vislcg) |
| hfst\_str | `str` | Analysis stream in the XFST/HFST format |
| replace\_tag | `None` | Replace a tag in this reading |
| to\_dict | `dict` | Convert to a `dict` object |

### `Tag` object

| Property | Type | Description |
| --- | --- | --- |
| name | `str` | The name of this tag |
| ms\_feat | `str` | Morphosyntactic feature that this tag is associated with (e.g. `Dat` has ms\_feat `CASE`) |
| detail | `str` | Description of the tag's purpose or meaning |
| is\_L2\_error | `bool` | Whether this tag indicates a second-language learner error |

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
diag = udar.diagnose_L2('Етот малчик говорит по-русски.')
print(diag == {'Err/L2_e2je': {'Етот'}, 'Err/L2_NoSS': {'малчик'}})
# True
```

#### tag\_info()

This function will look up the meaning of any tag used by the analyzer.

```python
print(udar.tag_info('Err/L2_ii'))
# L2 error: Failure to change ending ие to ии in +Sg+Loc or +Sg+Dat, e.g. к Марие, о кафетерие, о знание
```

### Using the transducers manually

The transducers come in two varieties: the `Analyzer` class and the `Generator`
class. For memory efficiency, I recommend using the `get_analyzer` and
`get_generator` functions, which ensure that each flavor of the transducers
remains a singleton in memory.

#### Analyzer

The `Analyzer` can be initialized with or without analyses for second-language
learner errors using the keyword `L2_errors`.

```python
analyzer = udar.get_analyzer()  # by default, L2_errors is False
L2_analyzer = udar.get_analyzer(L2_errors=True)
```

`Analyzer`s are callable. They take a token `str` and return a sequence of
reading/weight `tuple`s.

```python
raw_readings1 = analyzer('сло́ва')
print(raw_readings1)
# (('слово+N+Neu+Inan+Sg+Gen', 5.9755859375),)

raw_readings2 = analyzer('слова')
print(raw_readings2)
# (('слово+N+Neu+Inan+Pl+Acc', 5.9755859375), ('слово+N+Neu+Inan+Pl+Nom', 5.9755859375), ('слово+N+Neu+Inan+Sg+Gen', 5.9755859375))
```

#### Generator

The `Generator` can be initialized in three varieties: unstressed, stressed,
and phonetic.

```python
generator = udar.get_generator()  # unstressed by default
stressed_generator = udar.get_generator(stressed=True)
phonetic_generator = udar.get_generator(phonetic=True)
```

`Generator`s are callable. They take a `Reading` or raw reading `str` and
return a surface form.

```python
print(stressed_generator('слово+N+Neu+Inan+Pl+Nom'))
# слова́
```

## Working with `Token`s and `Readings`s

You can easily check if a morphosyntactic tag is in a `Token`, `Reading`,
or `Subreading` using `in`:
 
```python
token2 = udar.Token('слова', analyze=True)
print(token2)
# слова [слово_N_Neu_Inan_Pl_Acc  слово_N_Neu_Inan_Pl_Nom  слово_N_Neu_Inan_Sg_Gen]

print('Gen' in token2)  # do any of the readings include Genitive case?
# True

print('слово' in token2)  # does not work for lemmas; use `in Token.lemmas`
# False

print('слово' in token2.lemmas)
# True
```

You can make a filtered list of a `Token`'s readings using the following idiom:

```python
pl_readings = [reading for reading in token2 if 'Pl' in reading]
print(pl_readings)
# [Reading(слово+N+Neu+Inan+Pl+Acc, 5.975586, ), Reading(слово+N+Neu+Inan+Pl+Nom, 5.975586, )]
```

## Related projects

https://github.com/mikahama/uralicNLP
