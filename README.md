# UDAR(enie)

[![Actions Status](https://github.com/reynoldsnlp/udar/workflows/pytest/badge.svg)](https://github.com/reynoldsnlp/udar/actions)
[![codecov](https://codecov.io/gh/reynoldsnlp/udar/branch/master/graph/badge.svg)](https://codecov.io/gh/reynoldsnlp/udar)

**U**DAR **D**oes **A**ccented **R**ussian: A finite-state morphological
analyzer of Russian that handles stressed wordforms.

A python wrapper for the [Russian finite-state
transducer](https://victorio.uit.no/langtech/trunk/langs/rus/) described
originally in chapter 2 of [my dissertation](http://hdl.handle.net/10037/9685).

#### Feature requests, issues, and pull requests are welcome!

## Dependencies

For all features to be available, you should have `hfst` and `vislcg3`
installed as command-line utilities.

## Installation

For now, `udar` can be installed directly from this repository using `pip`:

```bash
$ python3 -m pip install --user git+https://github.com/reynoldsnlp/udar
```

## Introduction

The most common use-case is to use the `Text` constructor to automatically
tokenize and analyze a text. The `repr` is an `xfst`/`hfst` stream:

```python
import udar
text1 = udar.Text('Мы удивились простотой системы.')
text1
# Мы	мы+Pron+Pers+Pl1+Nom	0.0
# 
# удивились	удивиться+V+Perf+IV+Pst+MFN+Pl	5.078125
# 
# простотой	простота+N+Fem+Inan+Sg+Ins	4.2109375
# 
# системы	система+N+Fem+Inan+Pl+Acc	5.4296875
# системы	система+N+Fem+Inan+Pl+Nom	5.4296875
# системы	система+N+Fem+Inan+Sg+Gen	5.4296875
# 
# .	.+CLB	0.0
```

### `Text` methods

`Text` objects have convenient methods for adding stress or converting to
phonetic transcription.

```python
text1.stressify()
# 'Мы́ удиви́лись простото́й систе́мы.'
text1.phoneticize()
"Мы́ уд'ив'и́л'ис' пръстʌто́й с'ис'т'э́мы."
```

### Convenience functions

A number of functions are included, both for convenience, and to give concrete
examples for using the API. They can be found in the `convenience.py` file.

#### noun\_distractors()

This function generates all six cases of a given noun. If the given noun is
singular, then the function generates singular forms. If the given noun is
plural, then the function generates plural forms. Such a list can be used in a
multiple-choice exercise, hence the name `distractors`.

```python
sg_paradigm = udar.noun_distractors('словом')
sg_paradigm == {'сло́ву', 'сло́ве', 'сло́вом', 'сло́ва', 'сло́во'}
# True
pl_paradigm = udar.noun_distractors('словах')
pl_paradigm == {'слова́м', 'слова́', 'слова́х', 'слова́ми', 'сло́в'}
# True
```

If unstressed forms are desired, simply pass the argument `stressed=False`.

#### diagnose\_L2()

This function will take a text string as the argument, and will return a
dictionary of all the types of L2 errors in the text, along with examples of
the error.

```python
diag = udar.diagnose_L2('Мы разговаривали в кафетерие с Таной')
diag == {'Err/L2_ii': {'кафетерие'}, 'Err/L2_Pal': {'Таной'}}
# True
```

#### tag\_info()

This function will look up the meaning of any tag used by the analyzer.

```python
udar.tag_info('Err/L2_ii')
# 'L2 error: Failure to change ending ие to ии in +Sg+Loc or +Sg+Dat, e.g. к Марие, о кафетерие, о знание'
```

### Using the analyzer manually

The analyzer itself is the `Udar` class, which can be initialized as one of
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
token1
# сло́ва [слово_N_Neu_Inan_Sg_Gen]
token2 = analyzer.lookup('слова')
token2
# слова [слово_N_Neu_Inan_Pl_Acc  слово_N_Neu_Inan_Pl_Nom  слово_N_Neu_Inan_Sg_Gen]
```

## Working with `Token`s and `Readings`s

You can easily check if a lemma or morphosyntactic tag are in a `Token` or
`Reading` using `in`:
 
```python
token2
# слова [слово_N_Neu_Inan_Pl_Acc  слово_N_Neu_Inan_Pl_Nom  слово_N_Neu_Inan_Sg_Gen]
'Gen' in token2  # do any of the readings include Genitive case?
# True
'слово' in token2  # do any of the readings have the lemma 'слово'?
# True
'новый' in token2
# False
```

You can make a filtered list of a `Token`'s readings using the following idiom:

```python
pl_readings = [reading for reading in token2 if 'Pl' in reading]
pl_readings
# [слово_N_Neu_Inan_Pl_Acc  слово_N_Neu_Inan_Pl_Nom]
```

Grammatical analyses are parsed into the following objects:

* Tag
    * A part of speech or a morphosyntactic property
* Reading
    * Lemma (`str`) and a set of `Tag`s
* Token
    * Surface form (`str`) and a list of `Reading`s
* Text
    * List of `Token`s


`Tag`s can be looked up using the `tag_info()` function:

```python
udar.tag_info('Err/L2_NoFV')
# Err/L2_NoFV	L2 error: Lack of fleeting vowel where it should be inserted, e.g. окн (compare окон)
```

### Related projects

https://github.com/mikahama/uralicNLP
