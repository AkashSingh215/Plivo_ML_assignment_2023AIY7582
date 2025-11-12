import json
import re
from typing import List, Optional

from .rules import (
    generate_candidates,
    extract_emails_candidates,
    is_valid_email,
)
from .ranker_onnx import PseudoLikelihoodRanker


class PostProcessor:
    def __init__(
        self,
        names_lex_path: str,
        onnx_model_path: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 64,
    ):
        with open(names_lex_path, "r", encoding="utf-8") as fh:
            self.names_lex = [x.strip() for x in fh if x.strip()]
        # pass argument name `onnx_path` to match PseudoLikelihoodRanker signature
        self.ranker = PseudoLikelihoodRanker(
            onnx_path=onnx_model_path, device=device, max_length=max_length
        )

    def _insert_normalized_email_into_text(self, original: str, normalized_email: str) -> str:
        """
        Replace a likely email-like token in the original with the normalized email.
        If none found, append the normalized email to the end (safe fallback).
        """
        tokens = original.split()
        email_idx = None
        for i, tok in enumerate(tokens):
            if "@" in tok or re.search(
                r"(gmail|yahoo|outlook|hotmail|icloud|rediff|server|mail)\w*",
                tok,
                flags=re.I,
            ):
                email_idx = i
                break
        if email_idx is not None:
            tokens[email_idx] = normalized_email
            return " ".join(tokens)
        return original.rstrip() + " " + normalized_email

    def choose_best_with_shortcut(self, original_text: str, candidates: List[str]) -> str:
        """
        Prefer deterministic candidates (normalized email/₹/numeric) before invoking the ranker.
        If an email normalized from the original exists, prefer candidates containing it,
        else insert it back into the original.
        """
        # If original text contained an email-like chunk that rules normalized, use it
        email_cands = extract_emails_candidates(original_text)
        if email_cands:
            for e in email_cands:
                if is_valid_email(e):
                    # prefer candidate that already contains normalized email
                    for c in candidates:
                        if e in c:
                            return c
                    # otherwise insert normalized email into original
                    return self._insert_normalized_email_into_text(original_text, e)

        # prefer candidate that looks like email/phone/₹ before ranking
        for c in candidates:
            if "@" in c and "." in c:
                return c
            if c.strip().startswith("₹"):
                return c
            digits = re.sub(r"\D", "", c)
            if 7 <= len(digits) <= 15:
                return c

        # fallback to ranker (batch scoring inside ranker)
        return self.ranker.choose_best(candidates)

    def _ensure_trailing_punct(self, text: str) -> str:
        """
        Conservative trailing punctuation insertion:
        - If already ends with .!? or , -> keep as-is
        - Don't add punctuation if last token is a valid email or ends with a digit
        - If sentence looks like a question (starts with modal/ask words) add '?', else '.'
        """
        s = text.strip()
        if not s:
            return s
        if s[-1] in ".!?,":
            return s
        last = s.split()[-1]
        if is_valid_email(last) or re.fullmatch(r".*\d$", last):
            return s
        first = s.split()[0].lower() if s.split() else ""
        if first in (
            "can",
            "shall",
            "will",
            "could",
            "would",
            "is",
            "are",
            "do",
            "does",
            "did",
            "should",
            "please",
            "confirm",
            "reply",
        ):
            return s + "?"
        return s + "."

    def process_one(self, text: str) -> str:
        """
        Main entry for a single utterance:
        1) Quick email-only path (fast)
        2) Generate candidates (rules)
        3) Deterministic shortcuts
        4) Batch ranker choose_best
        5) Ensure trailing punctuation
        """
        # 1) Quick email-only path: if original contains a normalizable valid email, return quickly.
        email_cands = extract_emails_candidates(text)
        if email_cands:
            for e in email_cands:
                if is_valid_email(e):
                    best = self._insert_normalized_email_into_text(text, e)
                    return self._ensure_trailing_punct(best)

        # 2) Generate candidates from rules (names lexicon used inside)
        cands = generate_candidates(text, self.names_lex)

        # 3) Deterministic preference & ranking
        best = self.choose_best_with_shortcut(text, cands)

        # 4) Trailing punctuation
        best = self._ensure_trailing_punct(best)
        return best


def run_file(
    input_path: str,
    output_path: str,
    names_lex_path: str,
    onnx_model_path: str = None,
    device: str = "cpu",
    max_length: int = 64,
):
    pp = PostProcessor(
        names_lex_path, onnx_model_path=onnx_model_path, device=device, max_length=max_length
    )
    with open(input_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    out = []
    for r in rows:
        pred = pp.process_one(r["text"])
        out.append({"id": r["id"], "text": pred})
    with open(output_path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
