"""Part-of-speech tag"""

from collections import defaultdict
from typing import Dict
from typing import Union


__all__ = ['Tag', 'tag_dict']

ambiguous_tag_dict = {'AnIn': {'Anim', 'Inan'},
                      'IT': {'IV', 'TV'},
                      }
for ambig in list(ambiguous_tag_dict):
    for unambig in ambiguous_tag_dict[ambig]:
        ambiguous_tag_dict[unambig] = {ambig}
ambiguous_tag_dict = defaultdict(set, ambiguous_tag_dict)


class Tag:
    """Grammatical tag expressing a morphosyntactic or other value."""
    __slots__ = ['name', 'ms_feat', 'detail', 'is_L2', 'is_Err']

    def __init__(self, name: str, ms_feat: str, detail: str):
        self.name = name
        self.ms_feat = ms_feat
        self.detail = detail
        self.is_L2 = name.startswith('Err/L2')
        self.is_Err = name.startswith('Err')

    def __repr__(self):
        return f'Tag({self.name})'

    def __str__(self):
        return f'{self.name}'

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        """Exactly equivalent to Tag or str."""
        try:
            return self.name == other.name
        except AttributeError:
            return self.name == other

    def is_congruent_with(self, other):
        """Like __eq__, but allow loose matches, e.g. AnIn == Anim."""
        return self == other or self.name in ambiguous_tag_dict[other]

    def __hash__(self):
        return hash(self.name)

    def info(self):
        return f'{self.detail}'


_tags = [('A', 'POS', 'Adjective'),
         ('Abbr', 'POS', 'Abbreviation'),
         ('Adv', 'POS', 'Adverb'),
         ('CC', 'POS', 'Coordinating conjunction'),
         ('CS', 'POS', 'Subordinating conjunction'),
         ('Det', 'POS', 'Determiner'),
         ('Interj', 'POS', 'Interjection'),
         ('N', 'POS', 'Noun'),
         ('Num', 'POS', 'Numeral'),
         ('Paren', 'POS', 'Parenthetical вводное слово'),
         ('Pcle', 'POS', 'Particle'),
         ('Po', 'POS', 'Postposition (ради is the only postposition)'),
         ('Pr', 'POS', 'Preposition'),
         ('Pron', 'POS', 'Pronoun'),
         ('V', 'POS', 'Verb'),
         ('All', 'subPOS', 'All: весь'),
         ('Coll', 'subPOS', 'Collective numerals'),
         ('Def', 'subPOS', 'Definite'),
         ('Dem', 'subPOS', 'Demonstrative'),
         ('Indef', 'subPOS', 'Indefinite: кто-то, кто-нибудь, кто-либо, кое-кто, etc.'),  # noqa: E501
         ('Interr', 'subPOS', 'Interrogative: кто, что, какой, ли, etc.'),
         ('Neg', 'subPOS', 'Negative: никто, некого, etc.'),
         ('Pers', 'subPOS', 'Personal'),
         ('Pos', 'subPOS', 'Possessive, e.g. его, наш'),
         ('Prcnt', 'subPOS', 'Percent'),
         ('Prop', 'subPOS', 'Proper'),
         ('Recip', 'subPOS', 'Reciprocal: друг друга'),
         ('Refl', 'subPOS', 'Pronoun себя, possessive свой'),
         ('Rel', 'subPOS', 'Relativizer, e.g. который, где, как, куда, сколько, etc.'),  # noqa: E501
         ('Impf', 'ASPECT', 'Imperfective'),
         ('Perf', 'ASPECT', 'Perfective'),
         ('IV', 'TRANSITIVITY', 'Intransitive'),
         ('TV', 'TRANSITIVITY', 'Transitive (Zaliznjak does not mark trans-only, so transitive verbs all have both TV and IV)'),  # noqa: E501
         ('Inf', 'MOOD', 'Infinitive'),
         ('Imp', 'MOOD', 'Imperatives: 2nd person = читай, 1st person = прочитаем'),  # noqa: E501
         ('Pst', 'TENSE', 'Past'),
         ('Prs', 'TENSE', 'Present'),
         ('Fut', 'TENSE', 'Future'),
         ('Sg1', 'NUMBERPERSON', '1 person sg'),
         ('Sg2', 'NUMBERPERSON', '2 person sg'),
         ('Sg3', 'NUMBERPERSON', '3 person sg'),
         ('Pl1', 'NUMBERPERSON', '1 person pl'),
         ('Pl2', 'NUMBERPERSON', '2 person pl'),
         ('Pl3', 'NUMBERPERSON', '3 person pl'),
         ('PrsAct', 'PARTICIPLE', 'Present active participle (+PrsAct+Adv for verbal adverbs)'),  # noqa: E501
         ('PrsPss', 'PARTICIPLE', 'Present passive participle (+PstAct+Adv for the verbal adverbs)'),  # noqa: E501
         ('PstAct', 'PARTICIPLE', 'Past active participle'),
         ('PstPss', 'PARTICIPLE', 'Past passive participle'),
         ('Pass', 'VOICE', 'Passive'),
         ('Imprs', '???', 'Impersonal (cannot have explicit subject)'),
         ('Lxc', '???', 'Lexicalized (for participial forms)'),
         ('Der', 'DERIVATION', 'Derived (for participial forms)'),
         ('Der/PrsAct', 'DERIVATION', 'Derived (for participial forms)'),
         ('Der/PrsPss', 'DERIVATION', 'Derived (for participial forms)'),
         ('Der/PstAct', 'DERIVATION', 'Derived (for participial forms)'),
         ('Der/PstPss', 'DERIVATION', 'Derived (for participial forms)'),
         ('Msc', 'GENDER', 'Masculine gender'),
         ('Neu', 'GENDER', 'Neuter gender'),
         ('Fem', 'GENDER', 'Feminine gender'),
         ('MFN', 'GENDER', 'gender unspecifiable (pl tantum)'),
         ('Inan', 'ANIMACY', 'Inanimate'),
         ('Anim', 'ANIMACY', 'Animate'),
         ('AnIn', 'ANIMACY', 'ambivalent animacy for non-accusative modifiers'),  # noqa: E501
         ('Sem/Sur', 'SEMANTIC', 'Surname (фамилия)'),
         ('Sem/Pat', 'SEMANTIC', 'Patronymic'),
         ('Sem/Ant', 'SEMANTIC', 'Anthroponym/Given name'),
         ('Sem/Alt', 'SEMANTIC', 'Other name'),
         ('Sg', 'NUMBER', 'Singular'),
         ('Pl', 'NUMBER', 'Plural'),
         ('Nom', 'CASE', 'Nominative case'),
         ('Acc', 'CASE', 'Accusative case'),
         ('Gen', 'CASE', 'Genitive case'),
         ('Gen2', 'CASE', 'Genitive 2 case'),
         ('Loc', 'CASE', 'Prepositional case'),
         ('Loc2', 'CASE', 'Locative case (на полу, в снегу)'),
         ('Dat', 'CASE', 'Dative case'),
         ('Ins', 'CASE', 'Instrumental case'),
         ('Voc', 'CASE', 'Vocative case'),
         ('Count', '???', 'Count (for человек/людей or лет/годов, etc. also шага́/шара́/часа́/etc.)'),  # noqa: E501
         ('Ord', 'subPOS', 'Ordinal'),
         ('Cmpar', 'subPOS', 'Comparative'),
         ('Sint', 'SYNTACTIC', 'Synthetic comparative is possible, e.g. старее'),  # noqa: E501
         ('Pred', 'SYNTACTIC', '"Predicate", also used for short-form adjectives'),  # noqa: E501
         ('Cmpnd', 'SYNTACTIC', '"Compound", used for compounding adjectives, such as русско-английский'),  # noqa: E501
         ('Att', 'subPOS', 'Attenuative comparatives like получше, поновее, etc.'),  # noqa: E501
         ('SENT', 'PUNCT', 'Clause boundary'),
         ('COMMA', 'PUNCT', 'Comma'),
         ('DASH', 'PUNCT', 'Dash'),
         ('LQUOT', 'PUNCT', 'Left quotation'),
         ('RQUOT', 'PUNCT', 'Right quotation'),
         ('QUOT', 'PUNCT', '"Ambidextrous" quotation'),
         ('LPAR', 'PUNCT', 'Left parenthesis/bracket'),
         ('RPAR', 'PUNCT', 'Right parenthesis/bracket'),
         ('LEFT', 'PUNCT', 'Left bracket punctuation'),
         ('RIGHT', 'PUNCT', 'Right bracket punctuation'),
         ('Prb', 'CONFIDENCE', '+Prb(lematic): затруднительно - предположительно - нет'),  # noqa: E501
         ('Fac', 'CONFIDENCE', 'Facultative'),
         ('PObj', 'SYNTACTIC', 'Object of preposition (epenthetic н: него нее них) TODO FIX WITHFRAN merge PObj and Epenth > Epenth? Sandhi?'),  # noqa: E501
         ('Epenth', 'PHON', 'epenthesis on prepositions (о~об~обо or в~во)      TODO FIX WITHFRAN merge PObj and Epenth > Epenth? Sandhi?'),  # noqa: E501
         ('Leng', 'PHON', 'Lengthened доброй~доброю (marks less-canonical wordform that has more syllables)'),  # noqa: E501
         ('Elid', 'PHON', 'Elided (Иванович~Иваныч, новее~новей, чтобы~чтоб, или~иль, коли~коль)'),  # noqa: E501
         ('Use/NG', 'USE', 'Do not generate (used for apertium, etc.)'),
         ('Use/Obs', 'USE', 'Obsolete'),
         ('Use/Ant', 'USE', 'Antiquated "устаревшее"'),
         ('Err/Orth', 'ERROR', 'Substandard'),
         ('Err/L2_ii', 'L2ERROR', 'L2 error: Failure to change ending ие to ии in +Sg+Loc or +Sg+Dat, e.g. к Марие, о кафетерие, о знание'),  # noqa: E501
         ('Err/L2_Pal', 'L2ERROR', 'L2 error: Palatalization: failure to place soft-indicating symbol after soft stem, e.g. земла (compare земля)'),  # noqa: E501
         ('Err/L2_FV', 'L2ERROR', 'L2 error: Presence of fleeting vowel where it should be deleted, e.g. отеца (compare отца)'),  # noqa: E501
         ('Err/L2_NoFV', 'L2ERROR', 'L2 error: Lack of fleeting vowel where it should be inserted, e.g. окн (compare окон)'),  # noqa: E501
         ('Err/L2_SRo', 'L2ERROR', 'L2 error: Failure to change о to е after hushers and ц, e.g. Сашой (compare Сашей)'),  # noqa: E501
         ('Err/L2_SRy', 'L2ERROR', 'L2 error: Failure to change ы to и after hushers and velars, e.g. книгы (compare книги)'),  # noqa: E501
         ('Use/Aff', 'USE', 'Affectionate  ласкательное'),
         ('Use/Lit', 'USE', 'Literary  литературное'),
         ('Use/Flk', 'USE', 'folk poetry  народнопоэтическое'),
         ('Use/Dia', 'USE', 'Dialectal  областное'),
         ('Use/Poet', 'USE', 'Poetic  поэтический'),
         ('Use/Prof', 'USE', 'Professional   профессиональо'),
         ('Use/Anc', 'USE', 'Ancient   старинное'),
         ('Use/Aug', 'USE', 'Augmentative  увеличительное'),
         ('Use/Dim', 'USE', 'Diminutive  уменьшительное'),
         ('Use/Old', 'USE', 'Old (going out of date)  устаревающее'),
         ('Use/Anat', 'USE', 'Anatomical'),
         ('Use/Bio', 'USE', 'Biological'),
         ('Use/Bot', 'USE', 'Botanical'),
         ('Use/Geo', 'USE', 'Geological'),
         ('Use/Gram', 'USE', 'Grammatical'),
         ('Use/Zoo', 'USE', 'Zoological'),
         ('Use/Hist', 'USE', 'Historical'),
         ('Use/Math', 'USE', 'Mathematical'),
         ('Use/Med', 'USE', 'Medical'),
         ('Use/Mus', 'USE', 'Musical'),
         ('Use/Agr', 'USE', 'Agricultural'),
         ('Use/Tech', 'USE', 'Technical'),
         ('Use/Chem', 'USE', 'Chemical'),
         ('Use/Relig', 'USE', 'Church'),
         ('Use/Law', 'USE', 'Law'),
         ('Use/Mari', 'USE', 'Maritime'),
         ('Symbol', 'PUNCT', 'Symbol (independent symbols in the text stream, like £, €, ©)'),  # noqa: E501
         ('PUNCT', 'PUNCT', 'Punctuation'),
         ('CLB', 'SYNTAX', 'Clause boundary'),
        ]

tag_dict: Dict[Union[Tag, str], Tag] = {}
for tag_name, ms_feat, detail in _tags:
    if tag_name in tag_dict:
        raise NameError(f'{tag_name} is listed twice in _tags.')  # pragma: no cover  # noqa: E501
    tag = Tag(tag_name, ms_feat, detail)
    tag_dict[tag_name] = tag

ANIMACIES = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'ANIMACY']  # noqa: E501
ASPECTS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'ASPECT']  # noqa: E501
CASES = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'CASE']
ERRORS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat in {'ERROR', 'L2ERROR'}]  # noqa: E501
GENDERS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'GENDER']  # noqa: E501
L2ERRORS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'L2ERROR']  # noqa: E501
MOODS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'MOOD']
NUMBERS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'NUMBER']  # noqa: E501
PARTICIPLES = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'PARTICIPLE']  # noqa: E501
PERSONS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'NUMBERPERSON']  # noqa: E501
PUNCTUATIONS = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'PUNCT']  # noqa: E501
TENSES = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'TENSE']
TRANSITIVITIES = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'TRANSITIVITY']  # noqa: E501
USES = [tag.name for name, tag in tag_dict.items() if tag.ms_feat == 'USE']
