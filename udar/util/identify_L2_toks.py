import sys

import udar

if __name__ == '__main__':
    L2analyzer = udar.Udar('L2-analyzer')
    text = udar.Document(sys.stdin.read(), analyzer=L2analyzer)
    print(text)
    for tok in text:
        for r in tok.readings:
            if isinstance(r, udar.Reading):
                print(r, '\n\t', r.cg3_str(), file=sys.stderr)
        if tok.is_L2():
            print(tok, 'is an L2 error')
