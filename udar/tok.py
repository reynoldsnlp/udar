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

from .misc import destress
from .misc import Result
from .misc import StressParams
# from .reading import MultiReading
# from .reading import Reading
from .tag import Tag
from .transliterate import transliterate

if TYPE_CHECKING:
    import stanza  # type: ignore  # noqa: F401
    from .reading import MultiReading
    from .reading import Reading

__all__ = ['Token']

# declare combining stress characters for use in f-strings
ACUTE = '\u0301'
GRAVE = '\u0300'


class Token:
    """Custom token object"""
    __slots__ = ['_readings', '_stanza_tokens', 'annotation', 'deprel',
                 'end_char', 'features', 'head', 'id', 'lemmas', 'misc',
                 'phon_predictions', 'removed_readings', 'start_char',
                 'stress_ambig', 'stress_predictions', 'text', 'upper_indices',
                 'words']
    _readings: List['Reading']
    _stanza_tokens: List['stanza.models.common.doc.Token']
    annotation: str
    deprel: str
    end_char: int  # TODO
    features: Tuple
    head: int
    id: str  # 1-based index of word(s) in sentence (e.g., '4' or '7-9')
    lemmas: Set[str]
    misc: str
    phon_predictions: Dict[StressParams, set]
    removed_readings: List['Reading']
    start_char: int  # TODO
    stress_ambig: int  # number of stressed alternatives
    stress_predictions: Dict[StressParams, Tuple[str, Result]]
    text: str
    upper_indices: Set[int]  # TODO make hidden (add _)
    # words: List[Word]  # TODO

    def __init__(self, text, readings=[], removed_readings=[]):
        from .reading import _readify
        self._stanza_tokens = []
        self.annotation = ''
        self.features = ()
        self.removed_readings = [_readify(r) for r in removed_readings]
        self.text = text
        self.upper_indices = self.cap_indices()
        # keep self.readings last, since it calls readings.setter()
        self.readings = [_readify(r) for r in readings]

    @property
    def readings(self) -> List['Reading']:
        return self._readings

    @readings.setter
    def readings(self, readings):
        self._readings = [r for r in readings if r is not None]
        self.update_lemmas_stress_and_phon()

    def update_lemmas_stress_and_phon(self):
        self.lemmas = set()
        for r in self.readings:
            self.lemmas.add(r.lemma)
        self.phon_predictions = {}
        self.stress_predictions = {}
        self.stress_ambig = len(self.stresses())

    def __contains__(self, key: Union[str, Tag]):
        """Enable `in` Token. Checks both lemmas and tags."""
        if self.readings:
            return any(key in r for r in self.readings)
        else:
            return False

    def __repr__(self):
        return f'Token(text={self.text}, readings={self.readings!r}, removed_readings={self.removed_readings!r})'  # noqa: E501

    def __str__(self):
        return f'{self.text} [{"  ".join(str(r) for r in self.readings)}]'

    def hfst_str(self) -> str:
        """Token HFST-/XFST-style stream."""
        return '\n'.join(f'{self.text}\t{r.hfst_str()}\t{float(r.weight):.6f}'
                         for r in self.readings) \
               or f'{self.text}\t{self.text}+?\tinf'

    def cg3_str(self, traces=False, annotated=False) -> str:
        """Token CG3-style stream."""
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
        try:
            return (self.text == other.text
                    and len(self.readings) == len(other.readings)
                    and all(s == o for s, o in zip(sorted(self.readings),
                                                   sorted(other.readings))))
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self.text, self.readings))  # pragma: no cover

    def __len__(self):
        return len(self.readings)

    def __getitem__(self, i: int) -> 'Union[MultiReading, Reading]':
        # TODO tests
        return self.readings[i]

    def __iter__(self):
        return iter(self.readings)

    # def to_dict(self) -> List[str]:  # TODO
    #     return [r.to_dict() for r in self.readings]

    # def pretty_print(self):
    #     # TODO
    #     raise NotImplementedError

    @property
    def most_likely_reading(self) -> Optional['Reading']:
        """If one reading is marked as most likely, return it. Otherwise,
        select a most likely reading, label it as such, and return it.
        """
        most_likely = [r for r in self.readings if r.most_likely]
        if len(most_likely) == 1:
            return most_likely[0]
        else:
            try:
                max_weight = max(float(r.weight) for r in self.readings)
            except ValueError:
                return None
            most_likely_readings = [r for r in self.readings
                                    if float(r.weight) == max_weight]
            lucky_reading = choice(most_likely_readings)
            lucky_reading.most_likely = True
            return lucky_reading

    @property
    def most_likely_lemma(self) -> Optional[str]:
        """If one reading is marked as most likely, return its lemma.
        Otherwise, select a most likely reading, label it as such, and return
        its lemma.
        """
        try:
            return self.most_likely_reading.lemma  # type: ignore
        except AttributeError:
            return None

    def is_L2(self) -> bool:
        """Token: test if ALL readings contain an L2 error tag."""
        if self.readings:
            return all(r.L2_tags for r in self.readings)
        else:
            return False

    def has_L2(self) -> bool:
        """Token has ANY readings contain an L2 error tag."""
        if self.readings:
            return any(r.L2_tags for r in self.readings)
        else:
            return False

    def has_lemma(self, lemma: str) -> bool:
        """Token has ANY readings that contain the given lemma."""
        return lemma in self.lemmas

    def has_tag(self, tag: Union[Tag, str]) -> bool:
        """Token has ANY readings that contain the given tag."""
        # TODO do not need if...else here. any([]) is False.
        if self.readings:
            return any(tag in r for r in self.readings)
        else:
            return False

    def has_tag_in_most_likely_reading(self, tag: Union[Tag, str],
                                       partial=True) -> bool:
        """Token's most likely reading contains the given tag."""
        if self.readings:
            try:
                return tag in self.most_likely_reading  # type: ignore
            except TypeError:
                return False
        else:
            return False

    def cap_indices(self) -> Set[int]:
        """Token's indices of capitalized characters in the original."""
        return {i for i, char in enumerate(self.text) if char.isupper()}

    def recase(self, in_str: Optional[str]) -> Optional[str]:
        """Capitalize each letter in `in_str` indicated in `indices`."""
        if not self.upper_indices or in_str is None:
            return in_str
        grave_i = in_str.find(GRAVE)
        if grave_i == -1:
            grave_i = 255  # a small number bigger than the length of any word
        acute_i = in_str.find(ACUTE)
        if acute_i == -1:
            acute_i = 255
        return ''.join([char.upper()
                        if i + (i >= grave_i) + (i >= acute_i)  # True is 1
                        in self.upper_indices
                        else char
                        for i, char in enumerate(in_str)])

    def stresses(self, recase=True) -> Set[str]:
        """Return set of all surface forms from a Token's readings."""
        from .fsts import get_fst
        acc_gen = get_fst('acc-generator')
        if recase:
            try:
                stresses = {self.recase(r.generate(fst=acc_gen))
                            for r in self.readings}
            except AttributeError as e:  # pragma: no cover
                raise AttributeError('Problem generating stresses from: '
                                     f'{self} {self.readings}.') from e
        else:
            stresses = {r.generate(fst=acc_gen) for r in self.readings}
        stresses = {s for s in stresses if s is not None}
        return stresses  # type: ignore

    def stressed(self, disambiguated: bool = None, selection='safe',
                 guess=False, experiment=False, lemma=None) -> str:
        """Set of Token's surface forms with stress marked.

        disambiguated
            Boolean indicating whether the Token has undergone CG3 disamb.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            rand   -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.

        experiment
            1) Remove stress from Token.text
            2) Save prediction in each Token.stress_predictions[stress_params]

        lemma
            Limit readings to those with the given lemma.
        """
        if lemma:
            self.removed_readings.extend([r for r in self.readings
                                          if r.lemma != lemma])
            self.readings = [r for r in self.readings if r.lemma == lemma]
        stress_params = StressParams(disambiguated, selection, guess)
        stresses = self.stresses()
        if not stresses:
            if guess:
                pred = self.guess_syllable()
            elif experiment:
                pred = destress(self.text)
            else:
                pred = self.text
        elif len(stresses) == 1:
            pred = stresses.pop()
        else:
            if selection == 'safe':
                if experiment:
                    pred = destress(self.text)
                else:
                    pred = self.text
            elif selection == 'rand':
                pred = choice(list(stresses))
            elif selection == 'freq':
                raise NotImplementedError("The 'freq' selection method is not "
                                          'implemented yet.')
            elif selection == 'all':
                acutes = [(w.replace(GRAVE, '').index(ACUTE), ACUTE)
                          for w in stresses if ACUTE in w]
                graves = [(w.replace(ACUTE, '').index(GRAVE), GRAVE)
                          for w in stresses if GRAVE in w]
                yos = [(w.replace(GRAVE, '').replace(ACUTE, '').index('ё'), 'ё')  # noqa: E501
                       for w in stresses if 'ё' in w]
                positions = acutes + graves + yos
                word = list(destress(stresses.pop()))
                for i, char in sorted(positions, key=lambda x: (-x[0], x[1]),
                                      reverse=True):
                    if char in (ACUTE, GRAVE):
                        word.insert(i, char)
                    else:  # 'ё'
                        word[i] = char
                pred = ''.join(word)
            else:
                raise NotImplementedError(f"The '{selection}' selection "
                                          'method does not exist.')
        if experiment:
            self.stress_predictions[stress_params] = (pred,
                                                      self.stress_eval(pred))
        return pred

    def stress_eval(self, pred: str, ignore_monosyll=True) -> Result:
        """Token's stress prediction Result Enum value.

        If ignore_monosyll is True, then monosyllabic original forms always
        receive a score of SKIP. This is because many corpora make the (bad)
        assumption that all monosyllabic words are stressed.
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
        """Return set of all phonetic transcriptions from self.readings."""
        from .fsts import get_fst
        phon_gen = get_fst('phonetic-generator')
        phon_transcriptions = {r.generate(fst=phon_gen) for r in self.readings}
        return {t for t in phon_transcriptions if t is not None}

    def phonetic(self, disambiguated=None, selection='safe', guess=False,
                 experiment=False, lemma=None) -> str:
        """Token's phonetic transcription.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            rand   -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.
        """
        # TODO check if `all` selection is compatible with g2p.hfstol
        from .fsts import get_g2p
        g2p = get_g2p()
        if lemma:
            self.removed_readings.extend([r for r in self.readings
                                          if r.lemma != lemma])
            self.readings = [r for r in self.readings if r.lemma == lemma]
        transcriptions = self.phonetic_transcriptions()
        if not transcriptions:
            if guess:
                stress_pred = self.guess_syllable()
            elif experiment:
                stress_pred = destress(self.text)
            else:
                stress_pred = self.text
        elif len(transcriptions) == 1:
            return transcriptions.pop()
        else:  # if there are more than one possible transcription
            if selection == 'safe':
                if experiment:
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

    def phon_eval(self, pred: str):
        """Token Results of phonetic transcription predictions."""
        # raise NotImplementedError
        pass

    def guess(self, backoff=None) -> Optional[str]:
        if len(self.readings) > 0:
            random_reading = choice(self.readings)
            return random_reading.generate(fst='accented-generator')
        else:
            return self.guess_syllable()

    def guess_freq(self, backoff=None) -> str:
        """Guess stress location by selecting the most likely Reading based on
        frequency of Reading (backoff to frequency of tagset).

        Leftovers from dissertation script.
        """
        raise NotImplementedError()
        '''
        tag_freq_dict: Dict[str, int] = {}  # TODO import this dict
        lem_tag_freq_dict: Dict[str, int] = {}  # TODO import this dict
        read: Tuple[int, Optional[str]] = (0, None)
        tag: Tuple[int, Optional[str]] = (0, None)
        for r in self.readings:
            if '<' in r:
                tag_seq = r[r.index('<'):]
            else:
                tag_seq = ' '*16  # force tag backoff to fail if no '<'

            if tag_freq_dict.get(tag_seq, 0) > tag[0]:
                tag = (tag_freq_dict[tag_seq], r)

            if lem_tag_freq_dict.get(r, 0) > read[0]:
                read = (lem_tag_freq_dict[r], r)

        if read[0] > 0:
            return self.readings[read[1]][0]
        elif tag[0] > 0:
            return self.readings[tag[1]][0]
        else:
            if backoff is None:
                return self.text
            elif backoff == 'syllable':
                return self.guess_syllable()
            else:
                raise NotImplementedError('???')
        '''

    def guess_syllable(self) -> str:
        """Token: Place stress on the last vowel followed by a consonant.

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

    def transliterate(self, **kwargs) -> str:
        return transliterate(self.text, **kwargs)
