# UDAR(enie)

**U**DAR **D**oes **A**ccented **R**ussian: A finite-state morphological
analyzer of Russian that handles stressed wordforms.

A python wrapper for the Russian finite-state transducer described originally
in chapter 2 of [my dissertation](http://hdl.handle.net/10037/9685).

## Dependencies

For all features to be available, you should have `hfst` and `vislcg3`
installed as command-line utilities.

## Installation

For now, `udar` can be installed directly from this repository using `pip`:

```bash
$ python3 -m pip install --user git+https://github.com/reynoldsnlp/udar
```

## Introduction

The following objects are provided for convenience:

* Tag
    * A part of speech or a morphosyntactic property
* Reading
    * Lemma (`str`) and a set of `Tag`s
* Token
    * Surface form (`str`) and a list of `Reading`s
* Text
    * List of `Token`s

The `Udar` class has four flavors.

1. `analyzer`
1. `L2-analyzer`
1. `generator`
1. `accented-generator`

```python
>>> import udar
>>> analyzer = udar.Udar('analyzer')
```

The `Udar.lookup()` method takes a token `str` and returns a Token.

```python
>>> token1 = analyzer.lookup('сло́ва')
>>> 'Gen' in token1  # do any of the readings include Genitive case?
True
>>> print(token1)
(Token сло́ва [(Reading слово N Neu Inan Sg Gen)])
>>> token2 = analyzer.lookup('слова')
>>> print(token2)
(Token слова [(Reading слово N Neu Inan Pl Acc), (Reading слово N Neu Inan Pl Nom), (Reading слово N Neu Inan Sg Gen)])
```

The Text constructor automatically tokenizes and analyzes a text. The `repr` is
an `xfst`/`hfst` stream:

```python
>>> text1 = udar.Text('Мы удивились простотой системы.')
>>> text1
Мы	мы+Pron+Pers+Pl1+Nom	0.0

удивились	удивиться+V+Perf+IV+Pst+MFN+Pl	5.078125

простотой	простота+N+Fem+Inan+Sg+Ins	4.2109375

системы	система+N+Fem+Inan+Pl+Acc	5.4296875
системы	система+N+Fem+Inan+Pl+Nom	5.4296875
системы	система+N+Fem+Inan+Sg+Gen	5.4296875

.	.+CLB	0.0
```

Text objects have convenient functions, like returning the original text with
stress/accents.

```python
>>> text1.stressify()
'Мы́ удиви́лись простото́й систе́мы.'
```

Tags can be looked up using the `tag_info()` function:

```python
>>> tag_info('Err/L2_NoFV')
Err/L2_NoFV	L2 error: Lack of fleeting vowel where it should be inserted, e.g. окн (compare окон)
```
