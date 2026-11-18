"""Word n-gram language models with interpolated Kneser-Ney smoothing, for the Ear's beam search (decode.py). Our own
rather than KenLM: a few million words fit in Python dicts, and nothing needs building on the Spark.

Lower orders count how many different words come before an n-gram (continuation counts), except n-grams that start a
sentence; one discount per order, D = n1 / (n1 + 2 n2). Unknown words share the unigram floor.
"""
import collections
import functools
import math

BOS, EOS = "<s>", "</s>"


def discount(counts):
    n = collections.Counter(c for c in counts if c <= 2)
    return n[1] / (n[1] + 2 * n[2]) if n[1] else 0.5


class KneserNey:
    def __init__(self, sentences, order=4):
        """sentences: lists of normalised words (data.words)."""
        self.order, self.vocab = order, {"<unk>": 0, BOS: 1, EOS: 2}
        grams = [None] + [collections.Counter() for _ in range(order)]
        for s in sentences:
            ids = [1] + [self.vocab.setdefault(w, len(self.vocab)) for w in s] + [2]
            for n in range(1, order + 1):
                grams[n].update(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
        counts = {order: grams[order]}
        for n in range(order - 1, 0, -1):
            cont = collections.Counter(k[1:] for k in grams[n + 1])
            counts[n] = {k: c if k[0] == 1 else cont[k] for k, c in grams[n].items()}
            grams[n + 1] = None
        self.disc, self.gamma, self.size = {}, {}, {n: len(c) for n, c in counts.items()}
        for n in range(2, order + 1):
            d = discount(counts[n].values())
            denom, types = collections.Counter(), collections.Counter()
            for k, c in counts[n].items():
                denom[k[:-1]] += c
                types[k[:-1]] += 1
            for k, c in counts[n].items():
                self.disc[k] = (c - d) / denom[k[:-1]]
            for h, total in denom.items():
                self.gamma[h] = d * types[h] / total
            counts[n] = None
        uni = {k[0]: c for k, c in counts[1].items() if k[0] != 1}
        d, total = discount(uni.values()), sum(uni.values())
        self.uni = [d * len(uni) / total / (len(self.vocab) - 1)] * len(self.vocab)  # every word but <s>, <unk> too
        for w, c in uni.items():
            self.uni[w] += (c - d) / total
        self.unk_logp = math.log(self.uni[0])
        self.prefixes = {w[:i] for w in self.vocab for i in range(1, len(w) + 1)}  # for words still being spelled
        self.logp = functools.lru_cache(maxsize=1 << 20)(self._logp)

    def prob(self, ctx, word):
        """P(word | ctx); ctx = the preceding words, oldest first, starting with <s>."""
        w = self.vocab.get(word, 0)
        h = tuple(self.vocab.get(x, 0) for x in ctx[len(ctx) - self.order + 1:])
        p = self.uni[w]
        for n in range(1, len(h) + 1):
            g = self.gamma.get(h[-n:])
            if g is None:  # an unseen context: every longer one is unseen too
                break
            p = self.disc.get(h[-n:] + (w,), 0.0) + g * p
        return p

    def _logp(self, ctx, word):
        return math.log(self.prob(ctx, word))

    def coverage(self, sentences):
        """Share of the sentences' top-order n-grams that occur in the training text (a leak check)."""
        hit = total = 0
        for s in sentences:
            ids = [1] + [self.vocab.get(w, 0) for w in s] + [2]
            for i in range(len(ids) - self.order + 1):
                total += 1
                hit += tuple(ids[i:i + self.order]) in self.disc
        return hit / max(total, 1)


class Mix:
    """Linear interpolation of language models."""

    def __init__(self, models, weights):
        self.models, self.weights, self.order = models, weights, max(m.order for m in models)
        self.unk_logp = math.log(sum(w * math.exp(m.unk_logp) for m, w in zip(models, weights)))
        self.prefixes = set().union(*(m.prefixes for m in models))
        self.logp = functools.lru_cache(maxsize=1 << 20)(self._logp)

    def _logp(self, ctx, word):
        return math.log(sum(w * m.prob(ctx, word) for m, w in zip(self.models, self.weights)))


def perplexity(lm, sentences):
    total = n = 0
    for s in sentences:
        ctx = (BOS,)
        for w in s + [EOS]:
            total += lm.logp(ctx, w)
            n += 1
            ctx = (ctx + (w,))[1 - lm.order:]
    return math.exp(-total / n)
