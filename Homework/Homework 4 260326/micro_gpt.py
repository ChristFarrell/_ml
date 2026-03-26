"""
Micro GPT - A tiny GPT from scratch in pure Python, no dependencies.
Based on the concepts from Andrej Karpathy's microgpt.
"""

import math
import os
import random
import urllib.request

# =============================================================================
# 1. DATASET
# =============================================================================
# Download a simple dataset of names (one name per line)
if not os.path.exists('input.txt'):
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# =============================================================================
# 2. TOKENIZER
# =============================================================================
# Character-level tokenizer: each unique character gets an integer id
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)                          # Beginning of Sequence token
vocab_size = len(uchars) + 1               # +1 for BOS
print(f"vocab size: {vocab_size}")

# =============================================================================
# 3. AUTOGRAD ENGINE (Value class)
# =============================================================================
# Each Value wraps a scalar and tracks how it was computed.
# This is the core of backpropagation.

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        # Build topological order of computation graph
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

        # Backpropagate: apply chain rule in reverse order
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

# =============================================================================
# 4. MODEL PARAMETERS
# =============================================================================
# Hyperparameters
n_embd = 16            # embedding dimension
n_head = 4             # number of attention heads
n_layer = 1            # number of transformer layers
block_size = 12        # max sequence length
head_dim = n_embd // n_head

# Initialize weight matrices: each entry is a Value with a random Gaussian
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# Store all named parameter matrices
state_dict = {
    'wte': matrix(vocab_size, n_embd),      # token embedding
    'wpe': matrix(block_size, n_embd),       # position embedding
    'lm_head': matrix(vocab_size, n_embd),   # output projection
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Flatten all params into a single list for the optimizer
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# =============================================================================
# 5. MODEL ARCHITECTURE
# =============================================================================

def linear(x, w):
    """Matrix-vector multiply: y = W @ x"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    """Convert logits to probabilities"""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """Root Mean Square normalization"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    """
    Forward pass: process one token at position pos_id.
    Returns logits (scores) over the vocabulary.
    """
    # Token + position embedding
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        # --- Attention block ---
        x_residual = x
        x = rmsnorm(x)

        # Project to Q, K, V
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # Cache K, V for future positions
        keys[li].append(k)
        values[li].append(v)

        # Multi-head attention
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]

            # Attention scores: Q @ K^T / sqrt(d)
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)

            # Weighted sum of values
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        # Output projection + residual
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # --- MLP block ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    # Final projection to vocabulary
    logits = linear(x, state_dict['lm_head'])
    return logits

# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

# Adam optimizer settings
learning_rate = 0.008
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
m = [0.0] * len(params)   # first moment
v = [0.0] * len(params)   # second moment

num_steps = 500

print(f"\n--- training for {num_steps} steps ---\n")

for step in range(num_steps):
    # Pick a document, tokenize it, wrap with BOS
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward pass: build computation graph
    keys = [[] for _ in range(n_layer)]
    vals = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, vals)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()    # cross-entropy
        losses.append(loss_t)

    # Average loss over document
    loss = (1.0 / n) * sum(losses)

    # Backward pass
    loss.backward()

    # Adam optimizer update
    lr_t = learning_rate * (1 - step / num_steps)  # linear decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0.0

    print(f"step {step+1:4d} / {num_steps} | loss {loss.data:.4f}")

# =============================================================================
# 7. INFERENCE: Generate new names
# =============================================================================
temperature = 0.4

print(f"\n--- inference (temperature={temperature}) ---\n")

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
