# %% [markdown]
# # Autoregressive Word-Level Text Generation: SimpleRNN vs BiRNN vs LSTM vs GRU
# **Course:** Foundations of Generative AI (ICT4403) | **Assignment 2**
#
# **Dataset:** *Alice's Adventures in Wonderland* (Lewis Carroll), Project Gutenberg #11
# Source: https://www.gutenberg.org/ebooks/11
#
# **Pipeline:** load and clean text -> word-level tokenizer -> (input window, next word) pairs -> train 4 models under identical conditions -> evaluate -> generate text at 3 temperatures -> compare.
#
# Run: `Runtime > Change runtime type > GPU (T4)`, then `Runtime > Run all`.

# %% [markdown]
# ## 0. Setup and configuration

# %%
import os, re, time, random, urllib.request
from collections import Counter
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from IPython.display import display

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ---- One shared configuration: every model gets exactly the same settings ----
CFG = dict(
    seq_len=10,        # number of context words used to predict the next word
    min_freq=2,        # words rarer than this become <unk>
    val_frac=0.10,     # last 10% of the text is held out for validation
    embed_dim=128,
    rnn_units=128,     # hidden units of the recurrent layer (same for all models)
    dense_units=256,
    dropout=0.3,
    batch_size=128,
    max_epochs=30,
    patience=3,        # early stopping on validation loss
    lr=1e-3,
)
MODEL_NAMES = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]
os.makedirs("outputs", exist_ok=True)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# %% [markdown]
# ## 1. Dataset: download, clean, describe

# %%
DATA_URLS = [
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "https://www.gutenberg.org/files/11/11-0.txt",
]
DATA_PATH = "alice.txt"

if not os.path.exists(DATA_PATH):
    for url in DATA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8-sig")
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                f.write(raw)
            print("Downloaded:", url)
            break
        except Exception as e:
            print("Failed:", url, "->", e)

if not os.path.exists(DATA_PATH):
    raise RuntimeError("Download failed. Upload the book as 'alice.txt' via the Colab file panel and re-run.")

with open(DATA_PATH, encoding="utf-8") as f:
    raw_text = f.read()
print(f"Raw characters: {len(raw_text):,}")


# %%
def clean_gutenberg(text):
    """Remove the Project Gutenberg header/footer and normalise the text."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.I | re.S)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text, flags=re.I)
    if end:
        text = text[:end.start()]
    text = text.lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)   # curly apostrophes -> straight
    text = re.sub(r"\s+", " ", text)              # collapse whitespace/newlines
    return text.strip()


def tokenize(text):
    """Word-level tokens. Keeps words (with inner apostrophes) and . , ! ? ; : as separate tokens."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]", text)


clean_text = clean_gutenberg(raw_text)
tokens = tokenize(clean_text)
counts = Counter(tokens)

print(f"Total tokens      : {len(tokens):,}")
print(f"Unique tokens     : {len(counts):,}")
print(f"Avg word frequency: {len(tokens) / len(counts):.1f}")
print("Top 15 tokens     :", counts.most_common(15))
print("\nSample of cleaned text:\n", textwrap.fill(clean_text[:400], 90))

# %% [markdown]
# ## 2. Word-level tokenizer and autoregressive sequences
# Index 0 = `<pad>` (used to left-pad short seeds), index 1 = `<unk>` (rare words).
# For every position `i`, the **input** is words `[i, i+seq_len)` and the **target** is word `i+seq_len`.
# The text is split **first** (first 90% train, last 10% validation) and windows are built afterwards, so no window overlaps both sets.

# %%
itos = ["<pad>", "<unk>"] + sorted([w for w, c in counts.items() if c >= CFG["min_freq"]], key=lambda w: (-counts[w], w))
stoi = {w: i for i, w in enumerate(itos)}
VOCAB = len(itos)
L = CFG["seq_len"]

ids = np.array([stoi.get(w, 1) for w in tokens], dtype=np.int32)
unk_rate = (ids == 1).mean()
print(f"Vocabulary size: {VOCAB:,}  |  <unk> rate: {unk_rate:.2%}")


def make_xy(arr, seq_len):
    """Sliding window: X = seq_len words, y = the next word."""
    windows = np.lib.stride_tricks.sliding_window_view(arr, seq_len + 1)
    X = np.ascontiguousarray(windows[:, :-1])
    y = np.ascontiguousarray(windows[:, -1])
    return X, y


split = int(len(ids) * (1 - CFG["val_frac"]))
X_train, y_train = make_xy(ids[:split], L)
X_val, y_val = make_xy(ids[split:], L)
print(f"Train sequences: {len(X_train):,}  |  Validation sequences: {len(X_val):,}")

print("\nExample (input -> target):")
for k in [0, 1, 2]:
    print("  ", " ".join(itos[j] for j in X_train[k]), "->", itos[y_train[k]])


def make_ds(X, y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    return ds.batch(CFG["batch_size"]).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(X_train, y_train, shuffle=True)
val_ds = make_ds(X_val, y_val, shuffle=False)
train_eval_ds = make_ds(X_train, y_train, shuffle=False)  # un-shuffled copy for final evaluation

# %% [markdown]
# ## 3. Models
# All four share the same skeleton: `Embedding -> [recurrent layer] -> Dropout -> Dense(ReLU) -> Dropout -> Dense(softmax)`.
# Only the recurrent layer changes. Optimizer (Adam, lr 1e-3, grad-clip 1.0), batch size, loss, early stopping and seed are identical.
# Note: the Bidirectional layer outputs 2 x `rnn_units` features (forward + backward), which is inherent to the architecture.

# %%
def build_model(kind):
    tf.keras.utils.set_random_seed(SEED)   # same initial seed for every model
    inp = layers.Input(shape=(L,), dtype="int32")
    x = layers.Embedding(VOCAB, CFG["embed_dim"])(inp)

    if kind == "SimpleRNN":
        x = layers.SimpleRNN(CFG["rnn_units"])(x)
    elif kind == "BiRNN":
        x = layers.Bidirectional(layers.SimpleRNN(CFG["rnn_units"]))(x)
    elif kind == "LSTM":
        x = layers.LSTM(CFG["rnn_units"])(x)
    elif kind == "GRU":
        x = layers.GRU(CFG["rnn_units"])(x)
    else:
        raise ValueError(kind)

    x = layers.Dropout(CFG["dropout"])(x)
    x = layers.Dense(CFG["dense_units"], activation="relu")(x)
    x = layers.Dropout(CFG["dropout"])(x)
    out = layers.Dense(VOCAB, activation="softmax")(x)

    model = models.Model(inp, out, name=kind)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CFG["lr"], clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5_acc")],
    )
    return model


param_table = pd.DataFrame({n: [build_model(n).count_params()] for n in MODEL_NAMES}, index=["Parameters"]).T
display(param_table)

# %% [markdown]
# ## 4. Training (identical conditions, wall-clock time recorded)

# %%
class EpochTimer(callbacks.Callback):
    """Records the duration of each epoch."""
    def on_train_begin(self, logs=None):
        self.times = []
    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t)


histories, trained, rows = {}, {}, []

for name in MODEL_NAMES:
    print(f"\n{'=' * 20} Training {name} {'=' * 20}")
    model = build_model(name)
    timer = EpochTimer()
    stopper = callbacks.EarlyStopping(monitor="val_loss", patience=CFG["patience"], restore_best_weights=True)

    t0 = time.perf_counter()
    h = model.fit(train_ds, validation_data=val_ds, epochs=CFG["max_epochs"],
                  callbacks=[stopper, timer], verbose=2)
    total_time = time.perf_counter() - t0

    # Final metrics with best weights restored and dropout off
    tr_loss, tr_acc, _ = model.evaluate(train_eval_ds, verbose=0)
    va_loss, va_acc, va_top5 = model.evaluate(val_ds, verbose=0)
    best_ep = int(np.argmin(h.history["val_loss"])) + 1

    rows.append(dict(
        model=name,
        params=model.count_params(),
        epochs_run=len(h.history["loss"]),
        best_epoch=best_ep,
        train_loss=tr_loss,
        val_loss=va_loss,
        train_ppl=float(np.exp(tr_loss)),
        val_ppl=float(np.exp(va_loss)),
        train_acc=tr_acc,
        val_acc=va_acc,
        val_top5_acc=va_top5,
        train_time_s=total_time,
        time_to_best_s=sum(timer.times[:best_ep]),
        sec_per_epoch=float(np.mean(timer.times)),
    ))
    histories[name] = h.history
    trained[name] = model
    model.save(f"outputs/{name}.keras")

results = pd.DataFrame(rows).set_index("model")

# %% [markdown]
# ## 5. Quantitative evaluation
# Perplexity = exp(cross-entropy loss). Lower is better. A perplexity of P means the model is, on average, as uncertain as if it were choosing between P equally likely words.

# %%
fmt = {"train_loss": "{:.3f}", "val_loss": "{:.3f}", "train_ppl": "{:.1f}", "val_ppl": "{:.1f}",
       "train_acc": "{:.3%}", "val_acc": "{:.3%}", "val_top5_acc": "{:.3%}",
       "train_time_s": "{:.1f}", "time_to_best_s": "{:.1f}", "sec_per_epoch": "{:.2f}"}
display(results.style.format(fmt))
results.to_csv("outputs/results_table.csv")

# %%
colors = {"SimpleRNN": "tab:blue", "BiRNN": "tab:orange", "LSTM": "tab:green", "GRU": "tab:red"}

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
for n in MODEL_NAMES:
    ep = range(1, len(histories[n]["loss"]) + 1)
    ax[0].plot(ep, histories[n]["loss"], color=colors[n], label=n)
    ax[1].plot(ep, histories[n]["val_loss"], color=colors[n], label=n)
    ax[2].plot(ep, histories[n]["val_accuracy"], color=colors[n], label=n)
ax[0].set_title("Training loss");        ax[1].set_title("Validation loss");   ax[2].set_title("Validation accuracy")
for a in ax:
    a.set_xlabel("Epoch"); a.grid(alpha=0.3); a.legend()
ax[0].set_ylabel("Cross-entropy"); ax[2].set_ylabel("Top-1 accuracy")
plt.tight_layout(); plt.savefig("outputs/training_curves.png", dpi=150); plt.show()

fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for a, (col, title) in zip(ax, [("val_ppl", "Validation perplexity (lower is better)"),
                                ("val_acc", "Validation accuracy (higher is better)"),
                                ("val_top5_acc", "Validation top-5 accuracy"),
                                ("train_time_s", "Training time, seconds")]):
    a.bar(results.index, results[col], color=[colors[n] for n in results.index])
    a.set_title(title, fontsize=10); a.grid(axis="y", alpha=0.3)
    for i, v in enumerate(results[col]):
        a.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
plt.tight_layout(); plt.savefig("outputs/metric_comparison.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Text generation with temperature sampling
# At each step the model outputs a probability distribution over the vocabulary. With temperature $T$:
#
# $$p_i^{(T)} = \frac{\exp(\log p_i / T)}{\sum_j \exp(\log p_j / T)}$$
#
# - $T < 1$ sharpens the distribution (safer, more repetitive)
# - $T = 1$ samples from the model as it is
# - $T > 1$ flattens the distribution (more varied, more errors)
#
# `<pad>` and `<unk>` are never sampled. Every model uses the **same seeds** and the **same random-number stream** at each temperature, so the comparison is fair.

# %%
def encode_seed(seed_text):
    ids_ = [stoi.get(w, 1) for w in tokenize(seed_text.lower())][-L:]
    return [0] * (L - len(ids_)) + ids_            # left-pad short seeds


def detokenize(words):
    text = " ".join(words)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)   # remove space before punctuation


def sample_next(probs, temperature, rng):
    logits = np.log(probs.astype("float64") + 1e-12) / temperature
    logits[[0, 1]] = -np.inf                       # never emit <pad> / <unk>
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate_text(model, seed_text, n_words=40, temperature=0.7, rng=None):
    """Autoregressive generation: predict a word, append it, slide the window, repeat."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    window = encode_seed(seed_text)
    generated = []
    for _ in range(n_words):
        probs = model(np.array([window], dtype="int32"), training=False).numpy()[0]
        nxt = sample_next(probs, temperature, rng)
        generated.append(itos[nxt])
        window = window[1:] + [nxt]
    return detokenize(tokenize(seed_text.lower()) + generated), generated


# Quick sanity check
demo_text, _ = generate_text(trained["GRU"], "alice was beginning to get very tired of", 25, 0.7)
print(demo_text)

# %% [markdown]
# ## 7. Qualitative metrics (automatic proxies)
# | Metric | What it measures | Better |
# |---|---|---|
# | `distinct1` / `distinct2` | share of unique words / unique word pairs (diversity) | higher |
# | `rep3` | share of repeated word triples (repetition / looping) | lower |
# | `bigram_valid` | share of generated word pairs that occur in the book (local coherence proxy) | higher |
#
# These are cheap proxies, not a human judgement. Read the printed samples as well.

# %%
corpus_bigrams = set(zip(tokens, tokens[1:]))


def text_metrics(words):
    bi = list(zip(words, words[1:]))
    tri = list(zip(words, words[1:], words[2:]))
    return dict(
        distinct1=len(set(words)) / len(words),
        distinct2=len(set(bi)) / len(bi),
        rep3=1 - len(set(tri)) / len(tri),
        bigram_valid=float(np.mean([b in corpus_bigrams for b in bi])),
    )


SEEDS = [
    "alice was beginning to get very tired of",
    "the queen of hearts said to the",
    "the white rabbit looked at his watch and",
]
TEMPS = [0.2, 0.7, 1.2]
N_WORDS = 50
N_SAMPLES = 3          # samples per (model, temperature, seed) used for the metrics

gen_rows, samples = [], {}
for T in TEMPS:
    for name in MODEL_NAMES:
        rng = np.random.default_rng(SEED)          # identical random stream for every model
        for seed_text in SEEDS:
            for s in range(N_SAMPLES):
                text, words = generate_text(trained[name], seed_text, N_WORDS, T, rng)
                samples[(name, T, seed_text, s)] = text
                gen_rows.append(dict(model=name, temperature=T, seed=seed_text, **text_metrics(words)))

gen_df = pd.DataFrame(gen_rows)
gen_summary = gen_df.groupby(["model", "temperature"])[["distinct1", "distinct2", "rep3", "bigram_valid"]].mean().round(3)
gen_summary.to_csv("outputs/generation_metrics.csv")
display(gen_summary)

# %%
# Print one sample per seed, per model, per temperature (also saved to a text file)
lines = []
for seed_text in SEEDS:
    for T in TEMPS:
        lines.append(f"\n##### SEED: '{seed_text}'   |   TEMPERATURE: {T}")
        for name in MODEL_NAMES:
            lines.append(f"[{name}] " + textwrap.fill(samples[(name, T, seed_text, 0)], 110, subsequent_indent="   "))
report = "\n".join(lines)
print(report)
with open("outputs/generated_samples.txt", "w", encoding="utf-8") as f:
    f.write(report)

# %%
# Effect of temperature on diversity, repetition and coherence
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for a, (col, title) in zip(ax, [("distinct1", "Distinct-1 (diversity, higher = more varied)"),
                                ("distinct2", "Distinct-2 (diversity)"),
                                ("rep3", "Repeated trigrams (lower = less repetition)"),
                                ("bigram_valid", "Bigram validity (higher = more coherent)")]):
    for name in MODEL_NAMES:
        vals = gen_summary.loc[name][col].reindex(TEMPS)
        a.plot(TEMPS, vals, marker="o", color=colors[name], label=name)
    a.set_title(title, fontsize=10); a.set_xlabel("Temperature"); a.set_xticks(TEMPS); a.grid(alpha=0.3); a.legend()
plt.tight_layout(); plt.savefig("outputs/temperature_analysis.png", dpi=150); plt.show()

# %% [markdown]
# ## 8. Temperature analysis (edit after reading your own tables, samples and plots)
#
# **Low temperature (0.2):** the output is the most grammatical and locally coherent (highest `bigram_valid`), but it is also the most repetitive. The model keeps choosing the single most likely word, so it falls into loops such as *"the queen of the queen of the ..."* (highest `rep3`, lowest `distinct`).
#
# **Medium temperature (0.7):** best balance. Phrases still resemble the book's style, loops mostly disappear and diversity rises noticeably. This is the setting recommended for this dataset.
#
# **High temperature (1.2):** most diverse (highest `distinct1/2`, lowest `rep3`), but coherence drops: unusual word pairs appear, grammar breaks, and sentences drift off-topic. The weaker the model, the faster this happens.
#
# **Trade-off:** raising $T$ trades coherence for diversity. Repetition falls and local coherence falls as $T$ increases. Because the context window is only 10 words, none of the models can keep a long-range storyline; "contextual consistency" is limited to the last few words.

# %% [markdown]
# ## 9. Overall comparison and best model
# Models are ranked on five criteria (validation perplexity, validation accuracy, and at T = 0.7: distinct-2, repetition, bigram validity). Lowest average rank wins. Check this against the samples yourself before writing the conclusion.

# %%
q = gen_summary.xs(0.7, level="temperature")
rank_df = pd.DataFrame({
    "val_ppl":      results["val_ppl"].rank(ascending=True),
    "val_acc":      results["val_acc"].rank(ascending=False),
    "distinct2@0.7":   q["distinct2"].rank(ascending=False),
    "rep3@0.7":        q["rep3"].rank(ascending=True),
    "bigram_valid@0.7": q["bigram_valid"].rank(ascending=False),
})
rank_df["avg_rank"] = rank_df.mean(axis=1)
rank_df = rank_df.sort_values("avg_rank")
display(rank_df)
rank_df.to_csv("outputs/final_ranking.csv")

best_ppl = results["val_ppl"].idxmin()
best_overall = rank_df.index[0]
print(f"Lowest validation perplexity : {best_ppl} ({results.loc[best_ppl, 'val_ppl']:.1f})")
print(f"Best overall (average rank)  : {best_overall}")
print(f"Fastest per epoch            : {results['sec_per_epoch'].idxmin()}")

# %% [markdown]
# ## 10. Save everything for GitHub
# Downloads `outputs/` (plots, CSV tables, generated samples, saved models) as a zip.

# %%
import shutil
shutil.make_archive("fgai_assignment2_outputs", "zip", "outputs")
print("Created fgai_assignment2_outputs.zip")
try:
    from google.colab import files
    files.download("fgai_assignment2_outputs.zip")
except ImportError:
    print("Not running in Colab; find the zip in the current folder.")
