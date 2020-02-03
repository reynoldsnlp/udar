# UDAR(enie)

[![Actions Status](https://github.com/reynoldsnlp/udar/workflows/pytest/badge.svg)](https://github.com/reynoldsnlp/udar/actions)
[![codecov](https://codecov.io/gh/reynoldsnlp/udar/branch/master/graph/badge.svg)](https://codecov.io/gh/reynoldsnlp/udar)

**U**DAR **D**oes **A**ccented **R**ussian: A finite-state morphological
analyzer of Russian that handles stressed wordforms.

A python wrapper for the [Russian finite-state
transducer](https://victorio.uit.no/langtech/trunk/langs/rus/) described
originally in chapter 2 of [my dissertation](http://hdl.handle.net/10037/9685).

> Feature requests, issues, and pull requests are welcome!

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

`Text` objects have convenience functions, like returning the original text
with stress/accents.

```python
text1.stressify()
# 'Мы́ удиви́лись простото́й систе́мы.'
```

### objects and methods

The analyzer itself is the `Udar` class, which can be initialized as one of
four flavors:

1. `analyzer`: General-purpose analyzer
1. `L2-analyzer`: General analyzer with second-language learner errors added
1. `generator`: Generator of unstressed wordforms
1. `accented-generator`: Generator of stressed wordforms

```python
import udar
analyzer = udar.Udar('analyzer')
```

The `Udar.lookup()` method takes a token `str` and returns a `Token`.

```python
token1 = analyzer.lookup('сло́ва')
token1
# сло́ва [слово_N_Neu_Inan_Sg_Gen]
'Gen' in token1  # do any of the readings include Genitive case?
# True
token2 = analyzer.lookup('слова')
token2
# слова [слово_N_Neu_Inan_Pl_Acc  слово_N_Neu_Inan_Pl_Nom  слово_N_Neu_Inan_Sg_Gen]
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
