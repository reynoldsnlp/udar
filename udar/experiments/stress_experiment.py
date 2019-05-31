"""Using an arbitrary corpus, run an experiment to test accuracy of udar's
automatic stress annotation features."""

from collections import Counter
from pathlib import Path
import sys
from types import GeneratorType

import pandas as pd

import udar


class StressExperiment:
    """Experiment to test accuracy of udar's automatic stress annotation."""
    # TODO __slots__ = ['corpus', 'par_space', 'results']

    def __init__(self, corpus=None, par_space=None):
        """
        corpus -- filename or Path (or list of filenames/Paths)
                  If no corpus is given, text is taken directly from stdin.

        par_space -- list of StressParams
        """
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
        if par_space is None:
            self.par_space = [udar.StressParams(disambiguate, approach, guess)
                              for disambiguate in (False, True)
                              for approach in ('safe', 'random')
                              for guess in (False, True)]
        else:
            self.par_space = par_space
        self.results = None

    def __repr__(self):
        return 'StressExperiment(' + ', '.join(t.text_name
                                               for t in self.texts) + ')'

    def run(self):
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
                text.stressify(experiment=True, **kwargs)

    def param_eval(self, stress_params):
        """Combine metrics from individual texts for a specific parameter set.

        stress_params -- StressParams identifying the conditions to evaluate
        """
        print('Evaluating results for', stress_params, '...', file=sys.stderr)
        corp_results = Counter()
        for text in self.texts:
            text_results = text.stress_eval(stress_params)
            corp_results.update(text_results)
        corp_results['params'] = self.readable_name(stress_params)
        for k, v in stress_params._asdict().items():
            corp_results[k] = v
        return udar.compute_metrics(corp_results)

    def eval(self):
        for stress_params in self.par_space:
            self.param_eval(stress_params)

    @staticmethod
    def readable_name(stress_params):
        cg, selection, guess = stress_params
        cg = 'CG' if cg else 'noCG'
        guess = 'guess' if guess else 'no_guess'
        return '-'.join((cg, selection, guess))


if __name__ == '__main__':
    corpus = Path('stress_corpus').glob('*')
    exp = StressExperiment(corpus=corpus)
    exp.run()
    metrics = [exp.param_eval(sp) for sp in exp.par_space]
    df = pd.DataFrame(data=metrics)
    print(df)
