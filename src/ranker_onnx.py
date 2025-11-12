import os
import re
import numpy as np
from typing import List

try:
    import onnxruntime as ort
except Exception:
    ort = None

from transformers import AutoTokenizer

# Tunable knobs
CANDIDATE_CAP = 4
MAX_TOKENS = 64

# Tokenizer created once (cached)
_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def make_onnx_session(path: str, intra_threads: int = 1):
    if ort is None:
        raise RuntimeError("onnxruntime not installed")
    sess_opts = ort.SessionOptions()
    # enable optimizations
    try:
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass
    sess_opts.intra_op_num_threads = intra_threads
    sess_opts.inter_op_num_threads = 1
    sess_opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=sess_opts, providers=['CPUExecutionProvider'])


_re_email_and_tld = re.compile(r'@.*\.(com|in|co|net|org|edu|gov|io|ai)', flags=re.I)
_re_phone_like = re.compile(r'^[\d\-\s\(\)]+$')


def should_short_circuit(candidate: str) -> bool:
    """
    Deterministic cases where the rules output is already best:
    - contains email-like pattern with known TLD
    - starts with ₹ (currency)
    - phone / numeric-like sequence
    """
    if _re_email_and_tld.search(candidate):
        return True
    if candidate.strip().startswith("₹"):
        return True
    s = re.sub(r'\D', '', candidate)
    if 7 <= len(s) <= 15 and _re_phone_like.match(candidate.strip()):
        return True
    return False


def _batch_proxy_scores_from_logits(logits: np.ndarray, attention_mask: np.ndarray) -> List[float]:
    """
    logits: (B, L, V)
    attention_mask: (B, L)
    Proxy: for each token take max logit across V (proxy for confidence),
    zero out CLS/SEP and padded tokens, sum across tokens per example.
    """
    # logits may be float32
    B, L, V = logits.shape
    # zero out CLS and SEP per-example using attention_mask to find last token
    for i in range(B):
        # zero out first token
        logits[i, 0, :] = -1e9
        # determine last valid token index
        attn_sum = int(attention_mask[i].sum())
        last_idx = max(0, attn_sum - 1)
        if last_idx < L:
            logits[i, last_idx, :] = -1e9
    # max over vocab -> (B, L)
    max_per_token = logits.max(axis=2)
    # mask padded tokens
    masked = max_per_token * attention_mask
    # sum per example
    scores = masked.sum(axis=1).tolist()
    return scores


class PseudoLikelihoodRanker:
    def __init__(self, onnx_path: str = None, device: str = "cpu", max_length: int = MAX_TOKENS):
        self.tokenizer = _tokenizer
        self.onnx = None
        self.max_length = max_length
        if onnx_path:
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            self.onnx = make_onnx_session(onnx_path, intra_threads=1)
        else:
            raise RuntimeError("Please provide onnx_path to enable ONNX ranker.")

    def _score_batch(self, candidates: List[str]) -> List[float]:
        """
        Tokenize entire candidate list, run a single ONNX forward, return proxy scores.
        """
        toks = self.tokenizer(
            candidates,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        input_ids = toks["input_ids"].astype(np.int64)
        attention_mask = toks["attention_mask"].astype(np.float32)
        ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits = self.onnx.run(None, ort_inputs)[0]  # (B, L, V)
        # ensure float32 numpy arrays
        logits = np.asarray(logits, dtype=np.float32)
        attention_mask = np.asarray(attention_mask, dtype=np.float32)
        scores = _batch_proxy_scores_from_logits(logits, attention_mask)
        return scores

    def choose_best(self, candidates: List[str]) -> str:
        if not candidates:
            return ""
        cand = candidates[:CANDIDATE_CAP]
        # deterministic short-circuit: prefer valid email/phone/₹ candidates
        for c in cand:
            if should_short_circuit(c):
                return c
        # score remaining candidates in one batch
        try:
            scores = self._score_batch(cand)
        except Exception:
            # fallback: return first candidate deterministically if scoring fails
            return cand[0]
        best_idx = int(np.argmax(scores))
        return cand[best_idx]
