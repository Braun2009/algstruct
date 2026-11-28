"""LaTeX -> unicode notation, so every markdown renderer shows the formula.

A viewer that does not support a construct shows its source instead of the
formula, which is what inline `$\\rho(...)$` and `\\mathbf Z_t` did. Code spans and
fenced blocks render everywhere and read the same for this notation.

Inline math becomes a code span; display math becomes a fenced text block.

The scanner is recursive rather than a chain of substitutions: a chain glues
neighbours together (`a\\in` + `\\mathfrak{gl}` -> `a\\ingl`, and the macro reader
then sees one unknown name `ingl`).
"""
import pathlib
import re
import sys

BS = '\\'

GREEK = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π', 'rho': 'ρ',
    'sigma': 'σ', 'tau': 'τ', 'phi': 'φ', 'varphi': 'φ', 'chi': 'χ',
    'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ',
    'Pi': 'Π', 'Sigma': 'Σ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
}
SYMBOL = {
    'neq': '≠', 'ne': '≠', 'approx': '≈', 'sim': '∼', 'equiv': '≡',
    'leq': '≤', 'le': '≤', 'geq': '≥', 'ge': '≥', 'll': '≪', 'gg': '≫',
    'times': '×', 'cdot': '·', 'otimes': '⊗', 'oplus': '⊕', 'circ': '∘',
    'top': 'ᵀ', 'dagger': '†', 'pm': '±', 'mp': '∓',
    'to': '→', 'rightarrow': '→', 'mapsto': '↦', 'Rightarrow': '⇒',
    'leftarrow': '←', 'in': '∈', 'notin': '∉', 'subset': '⊂', 'supset': '⊃',
    'cup': '∪', 'cap': '∩', 'infty': '∞', 'partial': '∂', 'nabla': '∇',
    'sum': 'Σ', 'prod': '∏', 'int': '∫', 'langle': '⟨', 'rangle': '⟩',
    'lVert': '‖', 'rVert': '‖', 'Vert': '‖', 'vert': '|', '|': '‖',
    'cdots': '…', 'ldots': '…', 'dots': '…',
    'det': 'det', 'exp': 'exp', 'log': 'log', 'tr': 'tr', 'skew': 'skew',
    'sym': 'sym', 'max': 'max', 'min': 'min', 'sup': 'sup', 'inf': 'inf',
    'argmax': 'argmax', 'argmin': 'argmin', 'dim': 'dim', 'rank': 'rank',
    'curvearrowright': '↷', 'colon': ':', 'mid': '|', 'ast': '∗',
    'longmapsto': '⟼', 'cong': '≅', 'simeq': '≃', 'propto': '∝',
    'forall': '∀', 'exists': '∃', 'emptyset': '∅', 'subseteq': '⊆',
    'div': '÷', 'perp': '⊥', 'parallel': '∥',
}
ACCENT = {'overline': '\u0304', 'bar': '\u0304', 'hat': '\u0302',
          'tilde': '\u0303', 'dot': '\u0307', 'ddot': '\u0308',
          'underline': '\u0332', 'vec': '\u20d7'}
WRAP = {'mathbf', 'mathrm', 'mathcal', 'mathbb', 'mathfrak', 'mathsf',
        'operatorname', 'text', 'boldsymbol', 'mathit', 'boxed'}
DROP = {'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'displaystyle',
        'limits', 'nolimits'}
SPACE = {',': ' ', ';': ' ', ':': ' ', '!': '', ' ': ' ', 'quad': '   ',
         'qquad': '      '}
SUP = {'-1': '⁻¹', '-3': '⁻³', '2': '²', '3': '³', 'T': 'ᵀ', 't': 'ᵗ',
       'n': 'ⁿ', 'k': '_k', 'ᵀ': 'ᵀ'}
BIN = ('=', '≠', '≤', '≥', '≈', '→', '↦', '∈', '⇒', '≡', '·', '×', '⊗', '⊕', '+')


def _group(s, i):
    """Parse a brace group starting at s[i] == '{'. Returns (content, next_i)."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _token(s, i):
    """A single token argument: one char, or a \\command."""
    if i < len(s) and s[i] == '{':
        return _group(s, i)
    if i < len(s) and s[i] == BS:
        j = i + 1
        while j < len(s) and s[j].isalpha():
            j += 1
        return s[i:j], j
    return s[i:i + 1], i + 1


def conv(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == BS:
            j = i + 1
            while j < n and s[j].isalpha():
                j += 1
            name = s[i + 1:j]
            if not name:                       # \\ or \| or \, etc.
                sym = s[i + 1:i + 2]
                i += 2
                if sym == BS:
                    out.append('\n')
                elif sym in SYMBOL:
                    out.append(SYMBOL[sym])
                elif sym in (',', ';', ':', ' '):
                    out.append(' ')
                elif sym == '!':
                    pass
                elif sym in ('{', '}'):
                    out.append(sym)
                else:
                    out.append(sym)
                continue
            i = j
            if name in ACCENT:
                arg, i = _token(s, i)
                out.append(conv(arg) + ACCENT[name])
                continue
            if name in WRAP:
                arg, i = _token(s, i)
                out.append(conv(arg))
                continue
            if name in DROP:
                continue
            if name == 'frac':
                a, i = _token(s, i)
                b, i = _token(s, i)
                out.append(f'{conv(a)} / {conv(b)}')
                continue
            if name == 'sqrt':
                a, i = _token(s, i)
                out.append(f'√({conv(a)})')
                continue
            if name in SPACE:
                out.append(SPACE[name])
                continue
            if name in SYMBOL:
                out.append(SYMBOL[name])
                continue
            if name in GREEK:
                out.append(GREEK[name])
                continue
            out.append(BS + name)              # unknown: keep visible
            continue

        if c in '_^':
            arg, i = _token(s, i + 1)
            body = conv(arg)
            if c == '^' and body in SUP:
                out.append(SUP[body])
            elif len(body) == 1:
                out.append(c + body)
            else:
                out.append(c + '{' + body + '}')
            continue

        if c == '{':
            arg, i = _group(s, i)
            out.append(conv(arg))
            continue

        out.append(c)
        i += 1

    text = ''.join(out)
    return tidy(text)


def tidy(s):
    # \mathbf A_t\mathbf Z -> A_t Z: separate runs so subscripts do not fuse
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'([a-z0-9\)])([A-Z])', r'\1 \2', s)
    for op in BIN:
        s = re.sub(r'\s*' + re.escape(op) + r'\s*', ' ' + op + ' ', s)
    s = re.sub(r'\s*,\s*', ', ', s)
    for ch in ('‖', '⟨', '⟩'):
        s = re.sub(r'\s*' + re.escape(ch) + r'\s*', ch, s)
    s = re.sub(r'[ \t]+', ' ', s)
    return s.strip()


def process(text):
    out, i, n = [], 0, len(text)
    in_fence = False
    while i < n:
        if text.startswith('```', i):
            in_fence = not in_fence
            out.append('```')
            i += 3
            continue
        if in_fence:
            out.append(text[i])
            i += 1
            continue
        if text.startswith('$$', i):
            j = text.find('$$', i + 2)
            if j > 0:
                body = text[i + 2:j]
                parts = [p for p in re.split(r'\\\\(?![A-Za-z])', body)]
                lines = [conv(p) for p in parts]
                lines = [ln for ln in lines if ln]
                out.append('```text\n' + '\n'.join(lines) + '\n```')
                i = j + 2
                continue
        if text[i] == '$':
            j = text.find('$', i + 1)
            nl = text.find('\n', i + 1)
            if j > 0 and (nl < 0 or j < nl):
                inner = text[i + 1:j]
                if inner.strip() and '`' not in inner:
                    out.append('`' + conv(inner) + '`')
                    i = j + 1
                    continue
        out.append(text[i])
        i += 1
    return ''.join(out)


def main():
    write = '--write' in sys.argv
    files = ['README.md'] + sorted(str(p) for p in pathlib.Path('docs').glob('*.md'))
    # keep the authored LaTeX as the source of truth: run this after editing
    for f in files:
        p = pathlib.Path(f)
        src = p.read_text(encoding='utf-8')
        dst = process(src)
        print(f'{f}: {src.count("$")} $ -> {dst.count("$")} left')
        if write:
            p.write_text(dst, encoding='utf-8')
    if not write:
        print('(dry run; pass --write)')


if __name__ == '__main__':
    main()
