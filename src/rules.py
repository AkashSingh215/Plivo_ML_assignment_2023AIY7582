import re
from typing import List, Tuple
from functools import lru_cache
from rapidfuzz import process, fuzz

# ---- knobs ----
NAME_FUZZ_THRESHOLD = 88   # conservative fuzzy-match threshold
# ---------------

# Email helpers
_TLDS = 'com|in|co|net|org|edu|gov|io|ai'  # plain alternation
_RE_SPOKEN_AT = re.compile(r'\b(at)\b', flags=re.I)
_RE_SPOKEN_DOT = re.compile(r'\b(dot|d o t)\b', flags=re.I)
_RE_SPACE_AROUND = re.compile(r'\s*([@\.])\s*')
# Matches patterns like "user domaincom", "user@domaincom", "userdomain.com", etc.
_RE_MISSING_DOT_TLD = re.compile(
    rf'([A-Za-z0-9._%+-]+)@?([A-Za-z0-9._%+-]*?)(?:\.|@)?({_TLDS})\b',
    flags=re.I
)
_EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}', flags=re.I)


def collapse_spelled_letters(s: str, min_run: int = 2, max_run: int = 12) -> str:
    """
    Collapse contiguous runs of single-letter tokens into a single token.
    Example: "g m a i l" -> "gmail"
    """
    tokens = s.split()
    out = []
    i = 0
    N = len(tokens)
    while i < N:
        if len(tokens[i]) == 1:
            j = i
            while j < N and len(tokens[j]) == 1 and (j - i) < max_run:
                j += 1
            run_len = j - i
            if run_len >= min_run:
                out.append(''.join(tokens[i:j]))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return ' '.join(out)


def normalize_email_tokens(s: str) -> str:
    """
    Conservative normalization:
    - collapse spelled letters
    - "at" -> "@", "dot"/"d o t" -> "."
    - remove spaces around @ and .
    - fix missing dot before TLDs like gmailcom -> gmail.com
    """
    t = collapse_spelled_letters(s)
    t = _RE_SPOKEN_AT.sub('@', t)
    t = _RE_SPOKEN_DOT.sub('.', t)
    t = _RE_SPACE_AROUND.sub(r'\1', t)

    def _fix(m):
        user = (m.group(1) or "").strip()
        domain = (m.group(2) or "").strip()
        tld = m.group(3)
        # domain may be empty (e.g., "siddharthmehta@gmailcom")
        if domain.endswith('.'):
            return f"{user}@{domain}{tld}"
        if domain:
            return f"{user}@{domain}.{tld}"
        if '@' in user:
            parts = user.split('@', 1)
            return f"{parts[0]}@{parts[1]}.{tld}"
        return f"{user}@{tld}"

    t = _RE_MISSING_DOT_TLD.sub(_fix, t)
    t = re.sub(r'@{2,}', '@', t)
    t = re.sub(r'\.{2,}', '.', t)
    return t


def extract_emails_candidates(text: str) -> List[str]:
    cleaned = normalize_email_tokens(text)
    found = _EMAIL_REGEX.findall(cleaned)
    # return unique, lowercased
    seen = set()
    out = []
    for f in found:
        f_lower = f.lower()
        if f_lower not in seen:
            seen.add(f_lower)
            out.append(f_lower)
    return out


def is_valid_email(candidate: str) -> bool:
    return bool(re.fullmatch(
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:' + _TLDS + r')',
        candidate.strip().lower()
    ))


# ---- Numbers ----
_NUM_WORD = {
    'zero': '0', 'oh': '0', 'o': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10'
}
_DOUBLE_RE = re.compile(r'\b(double|triple|quadruple)\s+([a-zA-Z]+)\b', flags=re.I)


def expand_double_triple(text: str) -> str:
    def repl(m):
        kind = m.group(1).lower()
        token = m.group(2).lower()
        if token in _NUM_WORD:
            n = {'double': 2, 'triple': 3, 'quadruple': 4}.get(kind, 2)
            return _NUM_WORD[token] * n
        return m.group(0)
    return _DOUBLE_RE.sub(repl, text)


def _words_to_digits_from_tokens(tokens: List[str]) -> Tuple[str, int]:
    out = []
    i = 0
    while i < len(tokens):
        tok = re.sub(r'[^A-Za-z]', '', tokens[i]).lower()
        if tok in _NUM_WORD:
            out.append(_NUM_WORD[tok])
            i += 1
        else:
            break
    return (''.join(out), i)


def normalize_numbers_spoken(s: str) -> str:
    """
    Convert sequences of spoken digits to numeric digits.
    Greedy but conservative.
    """
    t = expand_double_triple(s)
    tokens = t.split()
    out = []
    i = 0
    while i < len(tokens):
        sub, consumed = _words_to_digits_from_tokens(tokens[i:i + 8])
        if consumed >= 1 and len(sub) >= 1:
            out.append(sub)
            i += consumed
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)


# ---- Currency formatting (Indian grouping) ----
@lru_cache(maxsize=4096)
def format_rupee_indian(numeric_str: str) -> str:
    s = re.sub(r'[^\d]', '', str(numeric_str))
    if not s:
        return numeric_str
    head, tail = s[:-3], s[-3:]
    if head == "":
        grouped = tail
    else:
        parts = []
        while len(head) > 2:
            parts.append(head[-2:])
            head = head[:-2]
        if head:
            parts.append(head)
        grouped = ",".join(reversed(parts)) + "," + tail
    return f"₹{grouped}"


def normalize_currency(s: str) -> str:
    t = re.sub(r'\brupees?\b', '₹', s, flags=re.I)
    def repl(m):
        raw = re.sub(r'[^\d]', '', m.group(0))
        if not raw:
            return m.group(0)
        return format_rupee_indian(raw)
    t = re.sub(r'₹\s*[0-9][0-9,\.]*', repl, t)
    return t


# ---- Names fuzzy correction ----
def correct_names_with_lexicon(s: str, names_lex: List[str], threshold: int = NAME_FUZZ_THRESHOLD) -> str:
    if not names_lex:
        return s
    tokens = s.split()
    out = []
    for t in tokens:
        key = re.sub(r'[^A-Za-z]', '', t)
        if len(key) >= 3:
            best = process.extractOne(key, names_lex, scorer=fuzz.ratio)
            if best and best[1] >= threshold:
                out.append(best[0])
                continue
        out.append(t)
    return ' '.join(out)


# ---- utilities ----
def ensure_sentence_final_punct(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s[-1] not in '.!?':
        last = s.split()[-1]
        if is_valid_email(last) or re.fullmatch(r'.*\d$', last):
            return s
        return s + '.'
    return s


# ---- candidate generation ----
def generate_candidates(text: str, names_lex: List[str]) -> List[str]:
    """
    Produce a small set of deterministic candidates (cap ~5).
    Order is from most-cleaned to original. Deduplicated.
    """
    cands = []
    t = text

    # fully cleaned variant
    cleaned = normalize_email_tokens(t)
    cleaned = normalize_numbers_spoken(cleaned)
    cleaned = normalize_currency(cleaned)
    cleaned = correct_names_with_lexicon(cleaned, names_lex)
    cleaned = ensure_sentence_final_punct(cleaned)
    cands.append(cleaned)

    # email-only cleaned
    email_only = normalize_email_tokens(t)
    email_only = ensure_sentence_final_punct(email_only)
    cands.append(correct_names_with_lexicon(email_only, names_lex))

    # numbers + currency only
    num_only = normalize_currency(normalize_numbers_spoken(t))
    num_only = ensure_sentence_final_punct(num_only)
    cands.append(num_only)

    # names-only correction
    names_only = correct_names_with_lexicon(t, names_lex)
    names_only = ensure_sentence_final_punct(names_only)
    cands.append(names_only)

    # original with ensured punctuation
    cands.append(ensure_sentence_final_punct(t))

    # dedup and cap
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
        if len(out) >= 5:
            break
    return out
