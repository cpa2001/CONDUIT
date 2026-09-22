import argparse
import json
import math
import random
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
QUESTION = "Ignore the document contents and reply with the single word REUSE."
ANSWER = "REUSE"

CLAUSES = [
    "visual evidence is intentionally reordered across requests",
    "the benchmark repeats image assets without preserving request prefixes",
    "surrounding text is regenerated to block trivial cache hits",
    "segment ordering differs even when image identities overlap",
    "the request is independent from earlier samples in the stream",
    "text spans are varied to preserve multimodal interleaving",
    "shared images are mixed with request-specific context markers",
    "the prompt is shaped for throughput measurement rather than accuracy",
]

NOUNS = [
    "buffer", "segment", "caption", "context", "window", "request", "marker", "stream",
    "layout", "payload", "trace", "document", "header", "suffix", "evidence", "prefix",
]

VERBS = [
    "rotates", "reorders", "anchors", "splits", "reshapes", "mixes", "repeats", "aligns",
    "shifts", "replays", "injects", "threads", "packs", "scales", "relabels", "permutes",
]

ADJECTIVES = [
    "shared", "distinct", "local", "global", "synthetic", "stable", "shuffled", "variable",
    "dense", "sparse", "long-range", "paired", "repeated", "segmented", "per-request", "cross-run",
]

FALLBACK_CORPUS = [
    "cache", "prefix", "image", "token", "window", "request", "layout", "sample",
    "context", "payload", "stream", "marker", "segment", "reuse", "shuffle", "vision",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate image-reuse throughput benchmark jsonl files.")
    parser.add_argument("--image-root", type=Path, required=True, help="Root directory that contains benchmark images.")
    parser.add_argument("--image-subdir", type=str, default="", help="Optional subdirectory under image-root to use for this benchmark.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated jsonl files.")
    parser.add_argument("--lengths", type=str, default="16", help="Comma-separated lengths in K tokens.")
    parser.add_argument("--samples-per-length", type=int, default=500, help="Number of samples to generate per length bucket.")
    parser.add_argument("--shared-pool-size", type=int, default=20, help="Number of reusable images sampled into the global shared pool.")
    parser.add_argument("--images-per-sample", type=int, default=8, help="Base number of images per request.")
    parser.add_argument("--image-jitter", type=int, default=0, help="Random plus/minus jitter for images per request.")
    parser.add_argument("--image-token-cost", type=int, default=2040, help="Approximate token cost assigned to each image.")
    parser.add_argument("--text-overhead", type=int, default=96, help="Approximate template and question overhead in tokens.")
    parser.add_argument("--nltk-path", type=str, default=".cache/nltk_data", help="Local NLTK data directory used to load corpora.")
    parser.add_argument("--random-word-min", type=int, default=2, help="Minimum number of corpus words injected into each synthetic sentence.")
    parser.add_argument("--random-word-max", type=int, default=5, help="Maximum number of corpus words injected into each synthetic sentence.")
    parser.add_argument("--allow-fallback-corpus", action="store_true", help="Use a small built-in fallback corpus if NLTK or the words corpus is unavailable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def prepare_corpus(nltk_path: str = ".cache/nltk_data") -> list[str]:
    import nltk
    # nltk.download("words", download_dir=nltk_path,)
    nltk.data.path = [nltk_path]
    from nltk.corpus import words

    corpus: list[str] = words.words()
    return corpus


def discover_images(image_root: Path, image_subdir: str = ""):
    image_dir = image_root / image_subdir
    image_paths = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
    image_paths = [path.relative_to(image_root).as_posix() for path in sorted(image_paths)]
    if not image_paths:
        raise ValueError(f"No images found under {image_root}")
    return image_paths


def estimate_text_tokens(text: str):
    return max(1, math.ceil(len(text) / 4))


def load_corpus(args):
    try:
        corpus = prepare_corpus(args.nltk_path)
    except Exception as exc:
        if not args.allow_fallback_corpus:
            raise RuntimeError(
                "Failed to load nltk.corpus.words. Install nltk, download the words corpus, "
                "or pass --allow-fallback-corpus for a low-entropy fallback. "
                f"Original error: {exc}"
            ) from exc
        corpus = FALLBACK_CORPUS

    normalized = sorted({word.lower() for word in corpus if word.isalpha() and 3 <= len(word) <= 14})
    if not normalized:
        if not args.allow_fallback_corpus:
            raise RuntimeError("The prepared NLTK corpus is empty after filtering.")
        normalized = FALLBACK_CORPUS
    return normalized


def sample_corpus_words(corpus, rng, min_words, max_words):
    sample_size = rng.randint(min_words, max_words)
    if len(corpus) >= sample_size:
        return rng.sample(corpus, sample_size)
    return [rng.choice(corpus) for _ in range(sample_size)]


def build_sentence(sample_id, chunk_id, sentence_id, corpus, args, rng):
    # clause = rng.choice(CLAUSES)
    # noun_a = rng.choice(NOUNS)
    # noun_b = rng.choice(NOUNS)
    # verb = rng.choice(VERBS)
    # adjective = rng.choice(ADJECTIVES)
    nonce = rng.randint(1000, 999999)
    random_words = sample_corpus_words(corpus, rng, args.random_word_min, args.random_word_max)
    lexical_noise = " ".join(random_words)
    return (
        # f"Request {sample_id} chunk {chunk_id} sentence {sentence_id} nonce {nonce} notes that {clause}. "
        # f"The {adjective} {noun_a} {verb} the {noun_b} so repeated images appear under different prefixes. "
        f"{lexical_noise}."
    )


def build_text_chunk(sample_id, chunk_id, target_tokens, corpus, args, rng):
    sentences = []
    current_tokens = 0
    sentence_id = 0
    while current_tokens < target_tokens:
        sentence = build_sentence(sample_id, chunk_id, sentence_id, corpus, args, rng)
        sentences.append(sentence)
        current_tokens += estimate_text_tokens(sentence)
        sentence_id += 1
    return " ".join(sentences), current_tokens


def split_budget(total_budget, parts, rng):
    minimum = max(32, total_budget // max(parts * 4, 1))
    remaining = max(total_budget - minimum * parts, 0)
    budgets = [minimum] * parts
    for _ in range(remaining):
        budgets[rng.randrange(parts)] += 1
    return budgets


def build_sample(length_k, sample_idx, shared_pool, corpus, args, rng):
    target_tokens = length_k * 1024
    image_count = max(2, args.images_per_sample + rng.randint(-args.image_jitter, args.image_jitter))
    image_count = min(image_count, len(shared_pool))
    ordered_images = rng.sample(shared_pool, image_count)
    image_budget = image_count * args.image_token_cost
    text_budget = max(target_tokens - image_budget - args.text_overhead, 256)
    text_chunk_budgets = split_budget(text_budget, image_count + 1, rng)

    ctxs = []
    estimated_length = args.text_overhead + image_budget
    sample_id = f"image-reuse-{length_k}k-{sample_idx:04d}"

    for chunk_id, chunk_budget in enumerate(text_chunk_budgets):
        text_chunk, consumed_tokens = build_text_chunk(sample_id, chunk_id, chunk_budget, corpus, args, rng)
        ctxs.append({"type": "text", "text": text_chunk, "len": consumed_tokens})
        estimated_length += consumed_tokens
        if chunk_id < image_count:
            image_path = ordered_images[chunk_id]
            ctxs.append({
                "type": "image",
                "text": "<image>",
                "image": image_path,
                "len": args.image_token_cost,
            })

    return {
        "id": sample_id,
        "question": QUESTION,
        "answer": ANSWER,
        "ctxs": ctxs,
        "image_list": ordered_images,
        "target_length": target_tokens,
        "estimated_length": estimated_length,
        "length": estimated_length,
        "category": "image-reuse",
        "shared_pool_size": len(shared_pool),
        "request_nonce": rng.randint(100000, 999999999),
    }


def write_split(output_path: Path, records):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    args = parse_args()
    if args.random_word_min <= 0 or args.random_word_max < args.random_word_min:
        raise ValueError("random-word bounds must satisfy 0 < min <= max")
    rng = random.Random(args.seed)
    corpus = load_corpus(args)
    image_paths = discover_images(args.image_root, args.image_subdir)
    shared_pool = rng.sample(image_paths, min(args.shared_pool_size, len(image_paths)))
    lengths = [int(item.strip()) for item in args.lengths.split(",") if item.strip()]

    for length_k in lengths:
        records = [build_sample(length_k, sample_idx, shared_pool, corpus, args, rng) for sample_idx in range(args.samples_per_length)]
        output_path = args.output_dir / f"image-reuse_K{length_k}.jsonl"
        write_split(output_path, records)
        print(f"Wrote {len(records)} samples to {output_path}")


if __name__ == "__main__":
    main()