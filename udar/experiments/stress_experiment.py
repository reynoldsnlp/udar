"""Using an arbitrary corpus, run an experiment to test accuracy of udar's
automatic stress annotation features."""

from collections import Counter
from pathlib import Path
import sys
from types import GeneratorType

import pandas as pd  # type: ignore

import udar


class StressExperiment:
    """Experiment to test accuracy of udar's automatic stress annotation."""
    __slots__ = ['corpus', 'par_space', 'results', 'texts']

    def __init__(self, corpus=None, par_space=None):
        """
        corpus -- filename or Path (or list of filenames/Paths)
                  If no corpus is given, text is taken directly from stdin.

        par_space -- list of StressParams or dict to pass to gen_param_space
        """
        print('Preparing corpus...', file=sys.stderr)
        self.corpus = corpus
        if corpus is None:
            if sys.stdin.isatty():
                print('No corpus given. Please try again.', file=sys.stderr)
                exit()
            else:
                self.texts = [udar.Text(sys.stdin.read(), text_name='stdin')]
        elif isinstance(corpus, str):
            with open(corpus) as f:
                self.texts = [udar.Text(f.read(), text_name=corpus)]
        elif isinstance(corpus, Path):
            self.texts = [udar.Text(corpus.read_text(), text_name=corpus.name)]
        elif isinstance(corpus, list) or isinstance(corpus, GeneratorType):
            corpus = list(corpus)
            if isinstance(corpus[0], str):
                self.texts = []
                for fname in corpus:
                    with open(fname) as f:
                        self.texts.append(udar.Text(f.read(), text_name=fname))
            elif isinstance(corpus[0], Path):
                self.texts = [udar.Text(p.read_text(), text_name=p.name)
                              for p in corpus]
        if par_space is None or isinstance(par_space, dict):
            self.par_space = self.gen_param_space(par_space)
        else:
            self.par_space = par_space
        self.results = None

    def __repr__(self):
        return f'StressExperiment(corpus={self.corpus}, par_space={self.par_space})'  # noqa: E501

    def run(self, tsvs=False):
        print('Annotating documents for each parameter set...',
              file=sys.stderr)
        for text in self.texts:
            print('\t', text.text_name, '...', file=sys.stderr)
            disambiguated = False
            for sp in sorted(self.par_space):
                print('\t\t', sp, file=sys.stderr)
                if sp.disambiguate and not disambiguated:
                    text.disambiguate()
                    disambiguated = True
                kwargs = sp._asdict()
                del kwargs['disambiguate']
                text.stressed(_experiment=True, **kwargs)
            if tsvs:
                text.stress_preds2tsv(path='tmp/')
        metrics = [self.param_eval(sp) for sp in self.par_space]
        self.results = pd.DataFrame(data=metrics)

    def param_eval(self, stress_params):
        """Combine metrics from individual texts for a specific parameter set.

        stress_params -- StressParams identifying the conditions to evaluate
        """
        print('Evaluating results for', stress_params, '...', file=sys.stderr)
        corp_results = Counter()
        for text in self.texts:
            text_results = text.stress_eval(stress_params)
            corp_results.update(text_results)
        corp_results['params'] = stress_params.readable_name()
        for k, v in stress_params._asdict().items():
            corp_results[k] = v
        return udar.compute_metrics(corp_results)

    def eval(self):
        for stress_params in self.par_space:
            self.param_eval(stress_params)

    @staticmethod
    def gen_param_space(in_dict=None):
        """Generate a list of StressParams made up of every possible
        combination of in_dict.values().

        in_dict -- key=parameter names, value=list of parameter values to test.
                   Keys of unimplemented parameters are ignored. If None is
                   given, the maximum parameter space is returned.
        """
        if in_dict is None:
            in_dict = {'disambiguate': [False, True],
                       'approach': ['safe', 'random'],
                       'guess': [False, True]}
        return [udar.StressParams(disambiguate, approach, guess)
                for disambiguate in sorted(in_dict['disambiguate'])
                for approach in in_dict['approach']
                for guess in in_dict['guess']]


if __name__ == '__main__':
    corpus = Path('stress_corpus').glob('*')
    # corpus = Path('RNC').glob('*')
    exp = StressExperiment(corpus=corpus)
    exp.run(tsvs=True)
    print(exp.results.to_csv(sep='\t', float_format='%.3f'))
