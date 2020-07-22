import sys

import udar

if __name__ == '__main__':
    L2analyzer = udar.Analyzer(L2_errors=True)
    text = udar.Document(sys.stdin.read(), analyzer=L2analyzer)
    print(text)
    for tok in text:
        for r in tok.readings:
            if isinstance(r, udar.Reading):
                print(r, '\n\t', r.cg3_str(), file=sys.stderr)
        if tok.is_L2_error():
            print(tok, 'is an L2 error')
