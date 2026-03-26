# Micro GPT

A tiny GPT model built from scratch in pure Python — no dependencies.

Based on the concepts from Andrej Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [How to Run](#how-to-run)
4. [Code Walkthrough](#code-walkthrough)
   - [Section 1: Dataset](#section-1-dataset)
   - [Section 2: Tokenizer](#section-2-tokenizer)
   - [Section 3: Autograd Engine (Value class)](#section-3-autograd-engine-value-class)
   - [Section 4: Model Parameters](#section-4-model-parameters)
   - [Section 5: Model Architecture](#section-5-model-architecture)
   - [Section 6: Training Loop](#section-6-training-loop)
   - [Section 7: Inference](#section-7-inference)
5. [Tuning Guide](#tuning-guide)
6. [The Big Picture](#the-big-picture)
7. [Project Structure](#project-structure)

---

## Project Overview

This is a **baby version of ChatGPT** in ~300 lines of pure Python.

| | ChatGPT | This Micro GPT |
|--|---------|----------------|
| **Data** | Trillions of tokens (internet) | 32,000 names |
| **Parameters** | Billions | 4,128 |
| **Training** | Months on GPU clusters | ~5 min on your CPU |
| **Algorithm** | Transformer GPT | Transformer GPT (same!) |

The algorithm is identical. Only scale differs.

---

## Requirements

- Python 3.8+
- Nothing else — no `pip install` needed

---

## How to Run

### Step 1: Open terminal in project folder

```bash
cd "D:\Farrell\_ml\Homework\Homework 04 260326"
```

### Step 2: Run the script

```bash
python micro_gpt.py
```

### Step 3: Wait for training

You will see output like:

```
num docs: 32033
vocab size: 27
num params: 4128

--- training for 500 steps ---

step    1 / 500 | loss 3.3544
step  100 / 500 | loss 2.8872
step  500 / 500 | loss 2.7590
```

### Step 4: See generated names

After training, the model generates new names:

```
--- inference (temperature=0.4) ---

  alin
  alan
  arina
  alina
  kana
  ...
```

---

## Code Walkthrough

Below is a detailed explanation of every section in `micro_gpt.py`.

---

### Section 1: Dataset

**Lines 11-21**

The model needs text data to learn from. We use a dataset of 32,000 baby names, one per line.

```python
if not os.path.exists('input.txt'):
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
```

**What it does:**

1. Downloads `input.txt` from GitHub if it doesn't exist locally
2. Reads the file, splits by newline, strips whitespace
3. Stores each name as a string in the `docs` list
4. Shuffles the names randomly so training sees them in random order

**Example data:**

```
docs = ["emma", "olivia", "ava", "isabella", "sophia", ...]
```

**Why shuffle?** If names are in alphabetical order, the model might learn the order instead of the actual patterns in names.

---

### Section 2: Tokenizer

**Lines 23-30**

Neural networks work with numbers, not text. We need to convert characters to integers and back.

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

**What it does:**

1. `''.join(docs)` — combines all names into one giant string
2. `set(...)` — gets unique characters only
3. `sorted(...)` — sorts them: `['a', 'b', 'c', ..., 'z']`
4. Each character gets an integer id based on its position

**Mapping table:**

```
'a' → 0
'b' → 1
'c' → 2
...
'z' → 25
BOS → 26  (special token: Beginning of Sequence)
```

**What is BOS?**

BOS (Beginning of Sequence) is a special token that acts as a fence between names. Each document is wrapped with BOS on both sides:

```
Input: "emma"
Tokens: [26, 4, 12, 12, 0, 26]
         BOS  e   m   m   a  BOS
```

The model learns:
- BOS at the start means "a name begins here"
- BOS at the end means "the name is done"

**Final values:**

```
vocab_size = 27  (26 letters + 1 BOS token)
```

---

### Section 3: Autograd Engine (Value class)

**Lines 32-109**

This is the most important section. The `Value` class is a mini version of what PyTorch does under the hood.

#### What is autograd?

When training a neural network, we need to know: **"If I nudge this parameter a little, does the loss go up or down, and by how much?"**

That number is called the **gradient**. Autograd computes it automatically.

#### The Value class

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data            # the actual number
        self.grad = 0.0             # gradient (computed during backward)
        self._children = children   # inputs that created this value
        self._local_grads = local_grads  # local derivatives
```

**Each `Value` stores:**

| Field | Purpose |
|-------|---------|
| `.data` | The actual number (e.g., 3.5) |
| `.grad` | How much the loss changes if this number changes |
| `._children` | The inputs that produced this value |
| `._local_grads` | The local derivative of this operation |

#### How operations work

Every math operation creates a new `Value` that remembers how it was computed:

```python
def __add__(self, other):
    return Value(self.data + other.data, (self, other), (1.0, 1.0))
    #           forward: a+b           inputs: a,b     local grads: d(a+b)/da=1, d(a+b)/db=1

def __mul__(self, other):
    return Value(self.data * other.data, (self, other), (other.data, self.data))
    #           forward: a*b           inputs: a,b     local grads: d(a*b)/da=b, d(a*b)/db=a
```

**Example:**

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c.data = 6.0, c._children = (a, b), c._local_grads = (3.0, 2.0)
```

The result `c` remembers: "I was made by multiplying `a` and `b`, and my local derivatives are 3.0 and 2.0."

#### Supported operations

| Operation | Code | Forward | Local gradient |
|-----------|------|---------|----------------|
| Addition | `a + b` | `a + b` | `∂/∂a = 1, ∂/∂b = 1` |
| Multiplication | `a * b` | `a * b` | `∂/∂a = b, ∂/∂b = a` |
| Power | `a ** n` | `a^n` | `∂/∂a = n * a^(n-1)` |
| Log | `a.log()` | `ln(a)` | `∂/∂a = 1/a` |
| Exp | `a.exp()` | `e^a` | `∂/∂a = e^a` |
| ReLU | `a.relu()` | `max(0, a)` | `∂/∂a = 1 if a>0, else 0` |

#### The backward() method

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)

    build_topo(self)
    self.grad = 1.0

    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

**Step by step:**

1. **Build topological order** — sort all nodes from inputs to output
2. **Set output gradient to 1.0** — because `dL/dL = 1`
3. **Walk backwards** — apply chain rule at each node

**Chain rule in simple terms:**

> "If a car is 2x faster than a bicycle, and the bicycle is 4x faster than a man, then the car is 2×4 = 8x faster than the man."

Same idea: multiply the local gradients along each path.

**Concrete example:**

```python
a = Value(2.0)
b = Value(3.0)
c = a * b       # c = 6.0
L = c + a       # L = 8.0
L.backward()

print(a.grad)   # 4.0  (dL/da = b + 1 = 3 + 1, via both paths)
print(b.grad)   # 2.0  (dL/db = a = 2)
```

The `+=` in `child.grad += ...` is important. When a value is used multiple times (like `a` in `c + a`), gradients from all paths are summed.

#### Why this matters

This `Value` class is the same algorithm that PyTorch's `loss.backward()` runs. The difference: PyTorch runs it on tensors (millions of numbers in parallel on GPU), we run it on scalars (one number at a time on CPU). The math is identical.

---

### Section 4: Model Parameters

**Lines 111-142**

The parameters are the "knowledge" of the model. They start random and get adjusted during training.

#### Hyperparameters

```python
n_embd = 16        # size of each token's vector representation
n_head = 4         # number of attention heads
n_layer = 1        # number of transformer layers
block_size = 12    # maximum sequence length
head_dim = n_embd // n_head  # 16 / 4 = 4 dimensions per head
```

#### Weight initialization

```python
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
```

Each weight is a `Value` initialized from a Gaussian (normal) distribution with mean=0 and std=0.08. Small random values prevent symmetry — if all weights were the same, all neurons would learn the same thing.

#### All parameters

```python
state_dict = {
    'wte':       matrix(27, 16),    # token embedding:      27 tokens × 16 dims
    'wpe':       matrix(12, 16),    # position embedding:   12 positions × 16 dims
    'lm_head':   matrix(27, 16),    # output projection:    27 tokens × 16 dims
}

# For each transformer layer:
state_dict['layer0.attn_wq'] = matrix(16, 16)   # query projection
state_dict['layer0.attn_wk'] = matrix(16, 16)   # key projection
state_dict['layer0.attn_wv'] = matrix(16, 16)   # value projection
state_dict['layer0.attn_wo'] = matrix(16, 16)   # output projection
state_dict['layer0.mlp_fc1'] = matrix(64, 16)   # MLP layer 1 (expand 4x)
state_dict['layer0.mlp_fc2'] = matrix(16, 64)   # MLP layer 2 (compress back)
```

**Parameter count breakdown:**

| Component | Shape | Count |
|-----------|-------|-------|
| Token embedding `wte` | 27 × 16 | 432 |
| Position embedding `wpe` | 12 × 16 | 192 |
| Attention Wq, Wk, Wv, Wo | 4 × (16 × 16) | 1,024 |
| MLP fc1 | 64 × 16 | 1,024 |
| MLP fc2 | 16 × 64 | 1,024 |
| Output `lm_head` | 27 × 16 | 432 |
| **Total** | | **4,128** |

---

### Section 5: Model Architecture

**Lines 144-229**

This is the GPT neural network. It processes one token at a time and outputs a probability distribution over the next token.

#### Helper functions

**linear(x, w)** — Matrix-vector multiply

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

Takes a vector `x` (length 16) and a weight matrix `w` (27 × 16), returns a new vector (length 27). Each output element is a dot product of one row of `w` with `x`.

**softmax(logits)** — Convert scores to probabilities

```python
def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

Takes 27 raw scores, converts them to 27 probabilities that sum to 1.0. Higher score → higher probability. We subtract `max_val` first to prevent overflow in `exp()`.

**rmsnorm(x)** — Root Mean Square normalization

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

Rescales a vector so its values don't grow or shrink as they flow through layers. This stabilizes training. `1e-5` is added to prevent division by zero.

#### The GPT forward pass

```python
def gpt(token_id, pos_id, keys, values):
```

This function processes **one token** at **one position** and returns 27 logits (one per vocabulary token).

**Step 1: Embeddings**

```python
tok_emb = state_dict['wte'][token_id]   # look up token vector
pos_emb = state_dict['wpe'][pos_id]     # look up position vector
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # combine them
x = rmsnorm(x)
```

The network can't process raw integers like `5`. It needs vectors. So:

- **Token embedding** (`wte`): converts token id to a 16-dimensional vector. Token "a" gets one vector, token "b" gets a different vector.
- **Position embedding** (`wpe`): converts position id to a 16-dimensional vector. Position 0 gets one vector, position 1 gets a different vector.
- **Add them together**: now `x` encodes both **what** the token is and **where** it is.

**Step 2: Transformer layer (attention + MLP)**

For each layer (we have 1):

```python
for li in range(n_layer):
```

**Step 2a: Multi-head attention**

```python
x_residual = x
x = rmsnorm(x)

q = linear(x, state_dict[f'layer{li}.attn_wq'])   # query: "what am I looking for?"
k = linear(x, state_dict[f'layer{li}.attn_wk'])   # key: "what do I contain?"
v = linear(x, state_dict[f'layer{li}.attn_wv'])   # value: "what do I offer?"
```

Attention is how tokens **communicate** with each other. The current token asks a question (query), and previous tokens answer (keys) and provide information (values).

```python
keys[li].append(k)
values[li].append(v)
```

We cache keys and values so the current token can attend to all previous tokens.

**Multi-head computation:**

```python
for h in range(n_head):          # 4 heads
    hs = h * head_dim             # head start index (0, 4, 8, 12)
    q_h = q[hs:hs + head_dim]    # slice query for this head
    k_h = [ki[hs:hs + head_dim] for ki in keys[li]]  # slice all cached keys
    v_h = [vi[hs:hs + head_dim] for vi in values[li]]

    # Attention score: how much should I attend to each previous token?
    attn_logits = [
        sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5
        for t in range(len(k_h))
    ]
    attn_weights = softmax(attn_logits)

    # Weighted sum: gather information from previous tokens
    head_out = [
        sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
        for j in range(head_dim)
    ]
    x_attn.extend(head_out)
```

**What each head does:**

1. Compute dot product between query and each cached key → attention scores
2. Divide by `sqrt(head_dim)` to prevent scores from getting too large
3. Apply softmax → attention weights (sum to 1.0)
4. Weighted sum of values → head output

Each head can learn to focus on different patterns. One head might look at vowels, another at consonant clusters, etc.

**Output projection + residual:**

```python
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
x = [a + b for a, b in zip(x, x_residual)]
```

The residual connection `x = x + attention(x)` lets gradients flow directly through the network, making deeper models trainable.

**Step 2b: MLP block**

```python
x_residual = x
x = rmsnorm(x)
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # expand to 4x size (16 → 64)
x = [xi.relu() for xi in x]                        # non-linearity
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # compress back (64 → 16)
x = [a + b for a, b in zip(x, x_residual)]         # residual
```

The MLP is where the model does its **thinking** per position. Unlike attention (which is about communication), MLP is purely local computation. The pattern: expand → activate → compress.

**Step 3: Output projection**

```python
logits = linear(x, state_dict['lm_head'])
return logits
```

The final hidden state (16 dims) is projected to vocabulary size (27 dims). Each of the 27 numbers is a "logit" — a raw score for how likely that token is to come next.

#### Architecture diagram

```
Token "e" (id=4) at position 1
         │
         ▼
  ┌─────────────┐
  │  Embedding   │  wte[4] + wpe[1] → 16-dim vector
  │  + RMSNorm   │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Attention   │  Q, K, V → 4 heads → concat → output proj
  │  + Residual  │  x = x + attention(x)
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │    MLP       │  expand 16→64 → ReLU → compress 64→16
  │  + Residual  │  x = x + mlp(x)
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  lm_head     │  16-dim → 27 logits
  └──────┬──────┘
         │
         ▼
  [0.1, -0.3, 0.8, ..., 0.2]  ← 27 scores, one per token
```

---

### Section 6: Training Loop

**Lines 231-280**

This is where the model learns. Each step: pick a name, predict next tokens, measure error, update parameters.

#### Adam optimizer setup

```python
learning_rate = 0.008
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
m = [0.0] * len(params)   # first moment (momentum)
v = [0.0] * len(params)   # second moment (adaptive learning rate)
```

**Why Adam instead of simple gradient descent?**

Simple gradient descent: `p -= lr * p.grad` (all parameters use same learning rate)

Adam: each parameter gets its own adaptive learning rate based on its gradient history. Parameters with consistently large gradients get smaller updates; parameters with small gradients get larger updates.

#### Training step

```python
for step in range(num_steps):

    # 1. Pick a name and tokenize it
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # 2. Forward pass: process each token, build computation graph
    keys = [[] for _ in range(n_layer)]
    vals = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, vals)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    # 3. Average loss over all positions
    loss = (1.0 / n) * sum(losses)

    # 4. Backward pass: compute gradients
    loss.backward()

    # 5. Adam update: adjust parameters
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0.0
```

**Example with "emma":**

```
Tokens: [BOS, e, m, m, a, BOS]

Position 0: input=BOS, target=e    → model should predict "e"
Position 1: input=e,   target=m    → model should predict "m"
Position 2: input=m,   target=m    → model should predict "m"
Position 3: input=m,   target=a    → model should predict "a"
Position 4: input=a,   target=BOS  → model should predict "BOS"
```

#### What is the loss?

**Cross-entropy loss** at each position: `loss = -log(probability of correct token)`

```
If model assigns 0.9 probability to correct token: loss = -log(0.9) = 0.105  (good)
If model assigns 0.1 probability to correct token: loss = -log(0.1) = 2.302  (bad)
If model assigns 0.01 probability to correct token: loss = -log(0.01) = 4.605 (very bad)
```

**Random guessing** among 27 tokens: `loss = -log(1/27) ≈ 3.30`

When training starts, loss is ~3.3 (random). As the model learns, loss decreases. Lower = better predictions.

#### What the Adam update does

```python
m[i] = beta1 * m[i] + (1 - beta1) * p.grad          # running average of gradient
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2      # running average of gradient²
m_hat = m[i] / (1 - beta1 ** (step + 1))              # bias correction
v_hat = v[i] / (1 - beta2 ** (step + 1))              # bias correction
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)    # update parameter
```

- `m` (first moment): momentum — if gradient is consistently positive, keep going that direction
- `v` (second moment): scales the update — parameters with large gradients get smaller steps
- `m_hat`, `v_hat`: bias correction for early steps when moments are still warming up
- `lr_t`: linear decay — learning rate starts at 0.008 and decreases to 0 by the end

---

### Section 7: Inference

**Lines 282-303**

After training, we generate new names by sampling from the model.

```python
temperature = 0.4

for i in range(20):
    keys = [[] for _ in range(n_layer)]
    vals = [[] for _ in range(n_layer)]
    token_id = BOS
    name = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, vals)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        name.append(uchars[token_id])

    print(f"  {''.join(name)}")
```

**How it works:**

1. Start with BOS token
2. Feed it through the model → get 27 logits
3. Divide logits by temperature
4. Convert to probabilities via softmax
5. Randomly sample one token based on probabilities
6. Feed sampled token back as input
7. Repeat until BOS or max length

#### What is temperature?

```
Temperature controls randomness:

  0.1  →  almost always picks the most likely token (boring, repetitive)
  0.4  →  picks likely tokens with some variation (our setting)
  1.0  →  samples directly from the learned distribution (balanced)
  2.0  →  flatter distribution, more random/creative (but less coherent)
```

**Mathematically:** dividing logits by temperature before softmax:
- Low temperature → probability differences get amplified → sharp distribution
- High temperature → probability differences get compressed → flat distribution

---

## Tuning Guide

Edit these values in `micro_gpt.py` to experiment:

```python
n_embd = 16        # embedding dimension (try: 8, 16, 32)
n_head = 4         # attention heads (must divide n_embd evenly)
n_layer = 1        # transformer layers (try: 1, 2)
block_size = 12    # max sequence length (try: 8, 12, 16)
num_steps = 500    # training steps (try: 200, 500, 1000)
temperature = 0.4  # generation randomness (0.1 to 1.0)
learning_rate = 0.008  # learning rate (try: 0.005 to 0.01)
```

**Quick combos:**

| Config | Params | Steps | Quality | Time |
|--------|--------|-------|---------|------|
| `n_embd=8, n_head=2, n_layer=1` | 1,296 | 200 | Basic | ~2 min |
| `n_embd=16, n_head=4, n_layer=1` | 4,128 | 500 | Better | ~5 min |
| `n_embd=16, n_head=4, n_layer=2` | ~8,000 | 1000 | Good | ~20 min |

> Note: Pure Python is slow. Larger models take much longer.

---

## The Big Picture

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                        │
│                                                          │
│  Dataset (names)                                         │
│      │                                                   │
│      ▼                                                   │
│  Tokenizer (char → id)                                   │
│      │                                                   │
│      ▼                                                   │
│  GPT Forward (token → logits → probabilities)            │
│      │                                                   │
│      ▼                                                   │
│  Loss = -log(prob of correct next token)                 │
│      │                                                   │
│      ▼                                                   │
│  Backward (autograd computes gradients)                  │
│      │                                                   │
│      ▼                                                   │
│  Adam Update (adjust parameters to reduce loss)          │
│      │                                                   │
│      └──── repeat 500 times ────────────────────────┐    │
│                                                     │    │
└─────────────────────────────────────────────────────┘    │
                                                          │
┌─────────────────────────────────────────────────────────┘
│                                                          │
│                    INFERENCE PHASE                        │
│                                                          │
│  Start with BOS token                                    │
│      │                                                   │
│      ▼                                                   │
│  GPT Forward → 27 logits                                 │
│      │                                                   │
│      ▼                                                   │
│  softmax(logits / temperature) → probabilities           │
│      │                                                   │
│      ▼                                                   │
│  Sample one token                                        │
│      │                                                   │
│      ▼                                                   │
│  Feed back as input → repeat                             │
│      │                                                   │
│      ▼                                                   │
│  Stop when BOS is generated                              │
│      │                                                   │
│      ▼                                                   │
│  Output: "arina" (a new hallucinated name)               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Homework 04 260326/
├── README.md          ← this file (detailed explanation)
├── micro_gpt.py       ← full implementation (~300 lines)
└── input.txt          ← dataset (auto-downloaded on first run)
```
