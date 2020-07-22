import argparse
import sys

from .document import Document


def parse_input(input_str: str, args: argparse.Namespace) -> Document:
    """Parse input string according to `args.input_type`."""
    if args.input_type == 'c':
        return Document.from_cg3(input_str)
    elif args.input_type == 'f':
        return Document.from_hfst(input_str)
    elif args.input_type == 'p':
        return Document(input_str, disambiguate=args.disambiguate)
    else:
        raise NotImplementedError


def print_output(doc: Document, args: argparse.Namespace):
    """Print output to stdout according to `args.output_type`."""
    if args.output_type == 'C':
        print(doc.cg3_str())
    elif args.output_type == 'F':
        print(doc)
    elif args.output_type == 'P':
        print(doc.stressed(selection=args.stress, guess=args.guess))
    elif args.output_type == 'T':
        print('\n'.join(tok.text for tok in doc))
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser(description='Perform morphological '
                                 'tokenization/analysis of input stream.')

in_group = parser.add_mutually_exclusive_group()
in_group.add_argument('-c', '--in-cg',
                      help='Set input format to CG',
                      action='store_const', dest='input_type', const='c')
in_group.add_argument('-f', '--in-fst',
                      help='Set input format to HFST/XFST',
                      action='store_const', dest='input_type', const='f')
in_group.add_argument('-p', '--in-plain',
                      help='Set input format to plain text',
                      action='store_const', dest='input_type', const='p')

out_group = parser.add_mutually_exclusive_group()
out_group.add_argument('-C', '--out-cg',
                       help='Set output format to CG',
                       action='store_const', dest='output_type', const='C')
out_group.add_argument('-F', '--out-fst',
                       help='Set output format to HFST/XFST',
                       action='store_const', dest='output_type', const='F')
out_group.add_argument('-P', '--out-plain',
                       help='Set output format to plain text (see -g and -s)',
                       action='store_const', dest='output_type', const='P')
out_group.add_argument('-T', '--out-tokens',
                       help='Set output format to tokens (1 token per line)',
                       action='store_const', dest='output_type', const='T')

parser.add_argument('-d', '--disambiguate', help='Apply constraint grammar',
                    action='store_true', default=False)
parser.add_argument('-g', '--guess-stress', help='For unknown tokens, apply '
                    'stress-guessing algorithm (Used in conjunction with -P)',
                    action='store_true', default=False)
parser.add_argument('-s', '--stress', help='How to select between ambiguous '
                    'stress possibilities (Used in conjunction with -P)',
                    choices=['safe', 'freq', 'rand', 'all', 'none'],
                    default='safe')
parser.add_argument('-v', '--verbose',
                    help='More extensive output (for debugging)',
                    action='count', default=0)
parser.add_argument('files', help='Input file(s). Use - for stdin (default).',
                    default=['-'], nargs='*')
parser.set_defaults(input_type='p', output_type='F')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.verbose:
        print(args, file=sys.stderr)

    if args.files == ['-']:
        files = [sys.stdin]
    else:
        files = [open(f) for f in args.files]

    for file in files:
        input_string = file.read()
        file.close()
        doc = parse_input(input_string, args)
        print_output(doc, args)
