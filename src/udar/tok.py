"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from random import choice
import re
import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from .fsts import get_analyzer
from .fsts import get_g2p
from .fsts import get_generator
from .misc import combine_stress
from .misc import destress
from .misc import Result
from .misc import StressParams
from .reading import Reading
from .tag import Tag
from .transliterate import transliterate

if TYPE_CHECKING:
    import stanza  # type: ignore  # noqa: F401

__all__ = ['Token']

# declare combining stress characters for use in f-strings
ACUTE = '\u0301'
GRAVE = '\u0300'


class Token:
    # TODO class docstring
    __slots__ = ['_readings', '_stanza_token', 'annotation', 'end_char',
                 'features', 'lemmas', 'misc', 'phon_predictions',
                 'removed_readings', 'start_char', 'stress_ambig',
                 'stress_predictions', 'text', '_upper_indices']
    _readings: List[Reading]
    _stanza_token: Optional['stanza.models.common.doc.Token']
    annotation: str
    end_char: int  # TODO
    features: Tuple
    lemmas: Set[str]
    misc: str
    phon_predictions: Dict[StressParams, set]
    removed_readings: List[Reading]
    start_char: int  # TODO
    stress_ambig: int  # number of stressed alternatives
    stress_predictions: Dict[StressParams, Tuple[str, Result]]
    text: str
    _upper_indices: Set[int]

    def __init__(self, text: str, *, _analyzer=None, analyze=False,
                 analyze_L2_errors=False,
                 readings: Union[List[Tuple[str, str, str]],
                                 List[Tuple[str, str]]] = None,
                 removed_readings: Union[List[Tuple[str, str, str]],
                                         List[Tuple[str, str]]] = None):
        """
        Parameters
        ----------

        text
            Original text of the token
        analyze
            (Optional) Whether to perform morphological analysis on the token
        analyze_L2_errors
            (Optional) Whether to include readings that describe learner
            errors. If ``analyze`` is False, this argument is ignored.
        readings
            (Optional) List of raw hfst readings, i.e. tuples. Tuples must have
            either 2 or 3 members: (lemma+tags, weight(, CG-rule)).
        removed_readings
            (Optional) List of raw hfst readings that have been removed,
            presumably by a constraint grammar. Tuples must have either 2 or 3
            members: (lemma+tags, weight(, CG-rule)).
        """
        self._stanza_token = None
        self.annotation = ''
        self.features = ()
        if removed_readings:
            self.removed_readings = [Reading(*r) for r in removed_readings]
        else:
            self.removed_readings = []
        self.text = text
        self._upper_indices = self._cap_indices()
        # keep self.readings last, since it calls readings.setter()
        if readings is not None:
            self.readings = [Reading(*r) for r in readings
                             if not r[0].endswith('?')]
        else:
            if _analyzer is not None:
                self.readings = [Reading(*r) for r in _analyzer(text)]
            elif analyze:
                _analyzer = get_analyzer(L2_errors=analyze_L2_errors)
                self.readings = [Reading(*r) for r in _analyzer(text)]
            else:
                self.readings = []

    @property
    def readings(self) -> List[Reading]:
        return self._readings

    @readings.setter
    def readings(self, readings: List[Reading]):
        self._readings = [r for r in readings if r is not None]
        self._update_lemmas_stress_and_phon()

    @property
    def deprel(self) -> str:
        if self._stanza_token is not None:
            return self._stanza_token.words[0].deprel
        else:
            return ''

    @property
    def head(self) -> int:
        if self._stanza_token is not None:
            return self._stanza_token.words[0].head
        else:
            return -1

    @property
    def id(self) -> str:
        if self._stanza_token is not None:
            # Assumes that all tokens are only 1 word (which is currently true)
            return next(iter(self._stanza_token.id))
        else:
            return ''

    def _update_lemmas_stress_and_phon(self):
        self.lemmas = set()
        for r in self.readings:
            self.lemmas.update(r.lemmas)
        self.phon_predictions = {}
        self.stress_predictions = {}
        self.stress_ambig = len(self.stresses())

    def __contains__(self, key: Union[str, Tag]):
        """Enable `in` Token."""
        return any(key in r for r in self.readings)

    def __repr__(self):
        return f'Token(text={self.text}, readings={self.readings!r}, removed_readings={self.removed_readings!r})'  # noqa: E501

    def __str__(self):
        return f'{self.text} [{"  ".join(str(r) for r in self.readings)}]'

    def hfst_str(self) -> str:
        """HFST-/XFST-style cohort."""
        return '\n'.join(f'{self.text}\t{r.hfst_str()}\t{float(r.weight):.6f}'
                         for r in self.readings) \
               or f'{self.text}\t{self.text}+?\tinf'

    def cg3_str(self, traces: bool = False, annotated: bool = False) -> str:
        """CG3-style cohort.

        Parameters
        ----------

        traces
            Whether to display removed readings (prefixed by ``;``), the same
            as would be returned by ``vislcg3 -t``.
        annotated
            Whether to add token annotations in the stream
        """
        output = '\n'.join(f'{r.cg3_str(traces=traces)}'
                           for r in self.readings) \
                 or f'\t"{self.text}" ? <W:{281474976710655.000000:.6f}>'
        if traces and self.removed_readings:
            removed = '\n'.join(f';{r.cg3_str(traces=traces)}'
                                for r in self.removed_readings)
            output = f'{output}\n{removed}'
        if annotated and self.annotation:
            ann = f'NB: ↓↓  {self.annotation}  ↓↓\n'
        else:
            ann = ''
        return f'{ann}"<{self.text}>"\n{output}'

    def __lt__(self, other):
        return ((self.text, self.readings, self.removed_readings)
                < (other.text, other.readings, other.removed_readings))

    def __eq__(self, other):
        # Do not include removed_readings in the comparison
        return (self.text == other.text
                and len(self.readings) == len(other.readings)
                and all(s == o for s, o in zip(sorted(self.readings),
                                               sorted(other.readings))))

    def __hash__(self):
        return hash((self.text, self.readings))  # pragma: no cover

    def __len__(self):
        return len(self.readings)

    def __getitem__(self, i: Union[int, slice]) -> Union[Reading,
                                                         List[Reading]]:
        # TODO tests
        return self.readings[i]

    def __iter__(self):
        return iter(self.readings)

    # def to_dict(self) -> List[str]:  # TODO
    #     return [r.to_dict() for r in self.readings]

    # def pretty_print(self):
    #     # TODO
    #     raise NotImplementedError

    def _filter_readings_using_stanza(self, readings=None) -> List[Reading]:
        """Return list of Readings that do not conflict with stanza's analysis
        of this token.
        """
        if readings is None:
            readings = self.readings
        if len(readings) < 2:
            return readings
        elif (self._stanza_token is not None
              and self._stanza_token.words[0].feats is not None):
            stanza_tags = set([self._stanza_token.words[0].upos])
            stanza_tags.update(re.findall(r'\w+=(\w+)\|?',
                                          self._stanza_token.words[0].feats))
            most_likely_readings = [r for r in readings
                                    if r.does_not_conflict(stanza_tags, 'UD')]
            if len(most_likely_readings) != 1:
                with open('/tmp/stanza_filter.log', 'a') as f:
                    print(self._stanza_token.words[0].feats,
                          ','.join(stanza_tags), readings, file=f)
            return most_likely_readings or readings
        else:
            return readings

    def _filter_readings_by_weight(self, readings=None) -> List[Reading]:
        """Return list of Readings with the highest weight value."""
        if readings is None:
            readings = self.readings
        try:
            max_weight = max(float(r.weight) for r in readings)
        except ValueError:
            return []
        else:
            return [r for r in readings if float(r.weight) == max_weight]

    def most_likely_reading(self, method=None) -> Optional[Reading]:
        """If one reading is marked as most likely, return it. Otherwise,
        select a most likely reading, label it as such, and return it.

        Parameters
        ----------

        method
            If most likely reading has not already been selected, which method
            to start from to select the most likely reading. If it has already
            been selected, then this argument is ignored. If the given method
            does not resolve all ambiguity, then subsequent methods are used
            until only one reading remains, in the order listed below.
            (default: ``weight``)

            * 'stanza' -- pick reading that is compatible with the Universal
              Dependency reading given by :py:mod:`stanza`.
            * 'weight' -- pick the reading with the highest weight value. If
              more than one reading shares the highest weight, randomly pick
              between them.
            * 'random' -- randomly select from readings
        """
        # TODO add 'reading_freq', 'tag_freq'
        if not self.readings:
            return None
        for r in self.readings:
            if r.is_most_likely:
                return r

        if method is None:
            method = 'weight'

        # TODO track which method resulted in final disambiguation?
        if method == 'stanza':
            readings = self._filter_readings_using_stanza()
            readings = self._filter_readings_by_weight(readings=readings)
            lucky_reading = choice(readings)
        elif method == 'weight':
            readings = self._filter_readings_by_weight()
            lucky_reading = choice(readings)
        elif method == 'random':
            lucky_reading = choice(self.readings)
        else:
            raise ValueError('`method` must be in {stanza, weight, random}.')
        lucky_reading.is_most_likely = True
        return lucky_reading

    def most_likely_lemmas(self, **kwargs) -> List[str]:
        r"""If one reading is marked as most likely, return list of its
        lemma(s). Otherwise, select a most likely reading, label it as such,
        and return list of its lemma(s).

        Parameters
        ----------

        \*\*kwargs
            The same arguments that can be passed to
            :py:meth:`most_likely_reading`
        """
        mlr = self.most_likely_reading(**kwargs)
        if mlr is not None:
            return mlr.lemmas
        else:
            return []

    def is_L2_error(self) -> bool:
        """Token: test if ALL readings contain an L2 error tag."""
        if self.readings:
            for r in self.readings:
                for subreading in r.subreadings:
                    if not any(tag.is_L2_error for tag in subreading):
                        return False
            return True
        else:
            return False

    def might_be_L2_error(self) -> bool:
        """Token has ANY readings that contain an L2 error tag."""
        return any(tag.is_L2_error for r in self.readings for tag in r)

    def has_tag_in_most_likely_reading(self, tag: Union[Tag, str],
                                       **kwargs) -> bool:
        r"""Token's most likely reading contains the given tag.

        Parameters
        ----------

        tag
            Tag (or name of tag) to check for
        \*\*kwargs
            The same arguments that can be passed to
            :py:meth:`most_likely_reading`
        """
        if self.readings:
            try:
                return tag in self.most_likely_reading(**kwargs)  # type: ignore  # noqa: E501
            except TypeError:  # if most_likely_reading returns None
                return False
        else:
            return False

    def _cap_indices(self) -> Set[int]:
        """Token's indices of capitalized characters in the original."""
        return {i for i, char in enumerate(self.text) if char.isupper()}

    def recase(self, in_str: Optional[str]) -> Optional[str]:
        """Capitalize each letter in ``in_str``, as indicated by
        ``self._upper_indices``.

        Parameters
        ----------

        in_str
            Input string to process, typically a generated wordform from the
            same paradigm as the readings of this Token.
        """
        if not self._upper_indices or in_str is None:
            return in_str
        grave_i = in_str.find(GRAVE)
        if grave_i == -1:
            grave_i = 255  # a small number bigger than the length of any word
        acute_i = in_str.find(ACUTE)
        if acute_i == -1:
            acute_i = 255
        return ''.join([char.upper()
                        if i + (i >= grave_i) + (i >= acute_i)  # True is 1
                        in self._upper_indices
                        else char
                        for i, char in enumerate(in_str)])

    def force_disambiguate(self, **kwargs):
        r"""Remove all ambiguity using one of the methods available in
        :py:meth:`Token.most_likely_reading`.

        Note that this method has no connection to the Constraint Grammar,
        which is implemented at the sentence level in
        :py:meth:`Sentence.disambiguate`.

        Parameters
        ----------

        \*\*kwargs
            The same arguments accepted by :py:meth:`Token.most_likely_reading`
        """
        mlr = self.most_likely_reading(**kwargs)
        if mlr is not None:
            self.readings = [mlr]
        else:
            self.readings = []

    def stresses(self, recase: bool = True) -> Set[str]:
        """Return set of all possible stressed forms, based on the Token's
        readings.

        Parameters
        ----------

        recase
            If ``True``, make each word match the capitalization of the
            original text (default: ``True``)
        """
        acc_gen = get_generator(stressed=True)
        if recase:
            try:
                stresses = {self.recase(r.generate(_generator=acc_gen))
                            for r in self.readings}
            except AttributeError as e:  # pragma: no cover
                raise AttributeError('Problem generating stresses from: '
                                     f'{self} {self.readings}.') from e
        else:
            stresses = {r.generate(_generator=acc_gen) for r in self.readings}
        stresses = {s for s in stresses if s is not None}
        return stresses  # type: ignore

    def stressed(self, *, selection: str = 'safe', guess: bool = False,
                 lemma: str = None, _experiment: bool = False,
                 _disambiguated: bool = None) -> str:
        """Predict a single stressed wordform for this Token.

        Parameters
        ----------
        selection
            Applies only to words in the lexicon

            * 'safe' (default) -- Only add stress if it is unambiguous.
            * 'freq' -- lemma+reading > lemma > reading (Not yet implemented)
            * 'rand' -- Randomly choose between specified stress positions.
            * 'all'  -- Add stress to all possible specified stress positions.
        guess
            If ``True``, make an "intelligent" guess for out-of-vocabulary
            words, i.e. when :py:attr:`self.readings` is empty.
        lemma
            (Optional) Limit readings to those with the given lemma.
        _experiment
            Used for evaluation of stress placement. If ``True``, do the
            following:
                1) Remove stress from Token.text
                2) Save prediction in Token.stress_predictions[stress_params]
        _disambiguated
            Used with experiments. Indicates whether the parent
            :py:class:`Sentence` has undergone CG3 disambiguation.
        """
        if lemma:
            self.removed_readings.extend([r for r in self.readings
                                          if lemma not in r.lemmas])
            self.readings = [r for r in self.readings if lemma in r.lemmas]
        stress_params = StressParams(_disambiguated, selection, guess)
        stresses = self.stresses()
        if not stresses:
            if guess:
                pred = self.guess_syllable()
            elif _experiment:
                pred = destress(self.text)
            else:
                pred = self.text
        elif len(stresses) == 1:
            pred = stresses.pop()
        else:
            if selection == 'safe':
                if _experiment:
                    pred = destress(self.text)
                else:
                    pred = self.text
            elif selection == 'rand':
                pred = choice(list(stresses))
            elif selection == 'freq':
                raise NotImplementedError("The 'freq' selection method is not "
                                          'implemented yet.')
            elif selection == 'all':
                pred = combine_stress(stresses)
            else:
                raise NotImplementedError(f"The '{selection}' selection "
                                          'method does not exist.')
        if _experiment:
            self.stress_predictions[stress_params] = (pred,
                                                      self.stress_eval(pred))
        return pred

    def guess_syllable(self) -> str:
        """Place stress on the last vowel followed by a consonant.

        This is a (bad) approximation of the last syllable of the stem. Not
        reliable at all, especially for forms with a consonant in the
        grammatical ending.
        """
        V = 'аэоуыяеёюи'
        if 'ё' in self.text or ACUTE in self.text:
            return self.text
        else:
            return re.sub(f'([{V}])([^{V}]+[{V}]+(?:[^{V}]+)?)$',
                          f'\\1{ACUTE}\\2',
                          self.text)

    def stress_eval(self, pred: str, ignore_monosyll: bool = True) -> Result:
        """Token's stress prediction Result Enum value.

        Parameters
        ----------

        pred
            Predicted wordform
        ignore_monosyll
            If True, then monosyllabic original forms always receive a score of
            SKIP. This is useful for evaluating against corpora that make the
            (bad) assumption that all monosyllabic words are stressed.
        """
        V = 'аэоуыяеёюи'
        if ignore_monosyll and len(re.findall(f'[{V}]', self.text)) < 2:
            return Result.SKIP
        try:
            orig_prim = {m.start()
                         for m in re.finditer(f'{ACUTE}|ё',
                                              self.text.replace(GRAVE, ''))}
            pred_prim = {m.start()
                         for m in re.finditer(f'{ACUTE}|ё',
                                              pred.replace(GRAVE, ''))}
        except AttributeError:
            print(f'WARN: unexpected pred type, text:{self.text} pred:{pred}',
                  self, file=sys.stderr)
            return Result.UNK
        both_prim = orig_prim.intersection(pred_prim)
        # orig_sec = {m.start() for m
        #             in re.finditer(GRAVE, self.text.replace(ACUTE, ''))}
        # pred_sec = {m.start() for m
        #             in re.finditer(GRAVE, pred.replace(ACUTE, ''))}
        # both_sec = orig_sec.intersection(pred_sec)
        if len(orig_prim) > 1 and len(pred_prim) > 1:
            print(f'Too many stress counts: {self.text}\t{pred}',
                  file=sys.stderr)
        if both_prim:  # if both share a primary stress mark
            return Result.TP
        elif pred_prim and orig_prim:
            return Result.FP
        elif pred_prim and not orig_prim:
            return Result.UNK
        elif not pred_prim and not orig_prim:
            return Result.TN
        elif not pred_prim and orig_prim:
            return Result.FN
        else:
            raise NotImplementedError(f'Bad Result: {orig_prim} {pred_prim}')

    def phonetic_transcriptions(self) -> Set[str]:
        """Return set of all phonetic transcriptions from
        :py:attr:`self.readings`.
        """
        phon_gen = get_generator(phonetic=True)
        phon_transcriptions = {r.generate(_generator=phon_gen)
                               for r in self.readings}
        return {t for t in phon_transcriptions if t is not None}

    def phonetic(self, *, selection: str = 'safe', guess: bool = False,
                 lemma: str = None, _experiment: bool = False,
                 _disambiguated: bool = None) -> str:
        """Predict the phonetic transcription of this Token, based on its
        readings.
        """
        # Parameters
        # ----------

        # selection
        #     If a token has readings that indicate different transcriptions,
        #     which method should be used to select which transcription to
        #     return. (applies only to words in the lexicon)
        #
        #     * 'safe' (default) -- Only add stress if it is unambiguous.
        #     * 'freq' -- lemma+reading > lemma > reading (Not yet implemented)
        #     * 'rand' -- Randomly choose between specified stress positions.
        #     * 'all'  -- Add stress to all possible specified stress positions

        # guess
        #     If ``True``, make an "intelligent" guess for out-of-lexicon words
        #     (i.e. when :py:attr:`self.readings` is empty).

        # lemma
        #     (Optional) Limit readings to those with the given lemma.

        # _experiment
        #     Used for evaluation of stress placement. If ``True``, do the
        #     following:
        #         1) Remove stress from Token.text
        #         2) Save prediction in Token.stress_predictions[stress_params]

        # _disambiguated
        #     Used with experiments. Indicates whether the parent
        #     :py:class:`Sentence` has undergone CG3 disambiguation.
        # """
        # TODO check if `all` selection is compatible with g2p.hfstol
        # TODO make this function suck less
        g2p = get_g2p()
        if lemma:
            self.removed_readings.extend([r for r in self.readings
                                          if lemma not in r.lemmas])
            self.readings = [r for r in self.readings if lemma in r.lemmas]
        transcriptions = self.phonetic_transcriptions()
        if not transcriptions:
            if guess:
                stress_pred = self.guess_syllable()
            elif _experiment:
                stress_pred = destress(self.text)
            else:
                stress_pred = self.text
        elif len(transcriptions) == 1:
            return transcriptions.pop()
        else:  # if there are more than one possible transcription
            if selection == 'safe':
                if _experiment:
                    stress_pred = destress(self.text)
                else:
                    stress_pred = self.text
            elif selection == 'rand':
                return choice(list(transcriptions))
            elif selection == 'freq':
                raise NotImplementedError("The 'freq' selection method is not "
                                          'implemented yet.')
            elif selection == 'all':
                stresses = self.stresses()
                acutes = [(w.replace(GRAVE, '').index(ACUTE), ACUTE)
                          for w in stresses if ACUTE in w]
                graves = [(w.replace(ACUTE, '').index(GRAVE), GRAVE)
                          for w in stresses if GRAVE in w]
                yos = [(w.replace(GRAVE, '').replace(ACUTE, '').index('ё'), 'ё')  # noqa: E501
                       for w in stresses if 'ё' in w]
                positions = acutes + graves + yos
                word = list(destress(transcriptions.pop()))
                for i, char in sorted(positions, key=lambda x: (-x[0], x[1]),
                                      reverse=True):
                    if char in (ACUTE, GRAVE):
                        word.insert(i, char)
                    else:  # 'ё'
                        word[i] = char
                stress_pred = ''.join(word)
            else:
                raise NotImplementedError(f"The '{selection}' selection "
                                          'method does not exist.')
        return g2p.lookup(stress_pred)[0][0]
        # TODO Are Y and S still implemented in g2p.twolc?
        # if out_token.endswith("я") or out_token.endswith("Я"):
        #     out_token += "Y"
        # elif out_token.endswith("ясь") or out_token.endswith("ЯСЬ"):
        #     out_token += "S"

    # def phon_eval(self, pred: str):
    #     """Results of phonetic transcription predictions."""
    #     # raise NotImplementedError
    #     pass

    def transliterate(self, **kwargs) -> str:
        r"""Transliterate original text to the latin alphabet.

        Parameters
        ----------

        \*\*kwargs
            All the same keyword arguments accepted by
            :py:func:`~udar.transliterate.transliterate`
        """
        return transliterate(self.text, **kwargs)

    def to_dict(self) -> Dict:
        """Convert to :py:obj:`dict`."""
        return {'id': self.id,
                'text': self.text,
                'readings': [r.to_dict() for r in self.readings],
                'removed_readings': [r.to_dict()
                                     for r in self.removed_readings],
                'head': self.head,
                'deprel': self.deprel,
                # 'misc': f'start_char={self.start_char}|end_char={self.end_char}',  # noqa: E501
                # 'ner': 'O'}
                }
