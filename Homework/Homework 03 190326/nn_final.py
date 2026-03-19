"""
Neural Network dengan AutoGrad, Adam, dan Gradient Descent
Implementasi Lengkap untuk Demo & Homework
"""

import math
import random
import numpy as np

# ============================================================
# 1. VALUE CLASS (AUTOGRAD ENGINE)
# ============================================================
class Value:
    """Unit komputasional dengan automatic differentiation"""
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # -------------------- Operasi Aritmatika --------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __neg__(self):
        return self * -1
    
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    # -------------------- Aktivasi --------------------
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        
        def _backward():
            self.grad += out.grad if out.data > 0 else 0
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(max(self.data, 1e-8)), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    # -------------------- Backward Pass --------------------
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# ============================================================
# 2. NEURAL NETWORK LAYERS
# ============================================================
class Neuron:
    """Single neuron dengan weighted sum + aktivasi"""
    
    def __init__(self, nin, activation='tanh'):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation
    
    def __call__(self, x):
        act = sum(w * x_i for w, x_i in zip(self.w, x)) + self.b
        
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        else:
            return act.tanh()
    
    def parameters(self):
        return self.w + [self.b]


class Layer:
    """Layer dengan banyak neuron"""
    
    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-Layer Perceptron"""
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], 'relu' if i < len(nouts)-1 else 'tanh')
            for i in range(len(nouts))
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ============================================================
# 3. OPTIMIZERS
# ============================================================
class SGD:
    """Stochastic Gradient Descent"""
    
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = [0.0] * len(params)
    
    def step(self):
        for i, p in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * p.grad
            p.data += self.velocities[i]
        self.zero_grad()
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class Adam:
    """Adam Optimizer (Adaptive Moment Estimation)"""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        self.zero_grad()
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0
    
    def get_state(self):
        return {'m': self.m.copy(), 'v': self.v.copy(), 't': self.t}


class AdamW:
    """Adam dengan Weight Decay (L2 regularization)"""
    
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
        self.zero_grad()
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


# ============================================================
# 4. LOSS FUNCTIONS
# ============================================================
def mse_loss(predictions, targets):
    """Mean Squared Error Loss"""
    loss = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    return loss


def cross_entropy_loss(logits, target_class):
    """Cross-Entropy Loss untuk klasifikasi"""
    exp_logits = [v.exp() for v in logits]
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    loss = -probs[target_class].log()
    return loss


def stable_cross_entropy(logits, target_class):
    """Numerically stable cross-entropy"""
    data_logits = [v.data for v in logits]
    max_logit = max(data_logits)
    shifted = [l - max_logit for l in data_logits]
    sum_exp = sum(math.exp(l) for l in shifted)
    log_sum_exp = math.log(sum_exp)
    
    out = Value(max_logit + log_sum_exp - data_logits[target_class])
    return out


# ============================================================
# 5. SOFTMAX TRANSFORMER
# ============================================================
def stable_softmax(logits):
    """Stable softmax untuk array Value"""
    data = [v.data for v in logits]
    max_val = max(data)
    exps = [math.exp(v - max_val) for v in data]
    sum_exps = sum(exps)
    return [Value(e / sum_exps) for e in exps]


class Attention:
    """Self-Attention Mechanism (Transformer Building Block)"""
    
    def __init__(self, d_model, n_heads=4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[Value(random.uniform(-0.1, 0.1)) for _ in range(d_model)] for _ in range(d_model)]
    
    def split_heads(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [[Value(x[i][j].data) for j in range(seq_len)] for i in range(batch_size)]
        return x
    
    def attention(self, Q, K, V):
        d_k = self.d_k
        scores = []
        for q in Q:
            row = []
            for k in K:
                dot = sum(q[i] * k[i] for i in range(d_k))
                row.append(dot / math.sqrt(d_k))
            scores.append(row)
        
        attn_weights = [[stable_softmax(row) for row in scores]]
        
        output = []
        for attn_row in attn_weights[0]:
            out_dim = [sum(attn_row[i] * V[j][i] for i in range(len(attn_row))) for j in range(self.d_model)]
            output.append(out_dim)
        
        return output
    
    def __call__(self, x):
        Q = [[sum(self.W_q[i][j] * x[k] for j, x_k in enumerate(x)) 
              for i in range(self.d_model)] for k in range(len(x))]
        K = [[sum(self.W_k[i][j] * x[k] for j, x_k in enumerate(x)) 
              for i in range(self.d_model)] for k in range(len(x))]
        V = [[sum(self.W_v[i][j] * x[k] for j, x_k in enumerate(x)) 
              for i in range(self.d_model)] for k in range(len(x))]
        
        attn_output = self.attention(Q, K, V)
        
        output = [[sum(self.W_o[i][j] * attn_output[k][i] for i in range(self.d_model)) 
                   for j in range(self.d_model)] for k in range(len(x))]
        
        return output
    
    def parameters(self):
        params = []
        for matrix in [self.W_q, self.W_k, self.W_v, self.W_o]:
            for row in matrix:
                params.extend(row)
        return params


# ============================================================
# 6. TRAINING LOOP
# ============================================================
def train(model, optimizer, data, targets, epochs, loss_fn=mse_loss, verbose=True):
    """Training loop generik"""
    history = []
    
    for epoch in range(epochs):
        predictions = [model(x) for x in data]
        loss = loss_fn(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        history.append({
            'epoch': epoch,
            'loss': loss.data,
            'predictions': [p.data if isinstance(p, Value) else p[0].data for p in predictions]
        })
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.data:.6f}")
    
    return history


# ============================================================
# DEMO: PERBANDINGAN OPTIMIZER
# ============================================================
def demo_comparison():
    """Demo perbandingan SGD vs Adam vs AdamW"""
    print("=" * 60)
    print("DEMO: Perbandingan Optimizer")
    print("=" * 60)
    
    def run_optimizer(Optimizer, name, lr=0.1):
        print(f"\n--- {name} (lr={lr}) ---")
        
        x = [Value(2.0)]
        w = [Value(0.5)]
        b = [Value(0.0)]
        params = w + b
        
        if Optimizer == Adam:
            opt = Optimizer(params, lr=lr)
        elif Optimizer == AdamW:
            opt = Optimizer(params, lr=lr)
        else:
            opt = Optimizer(params, lr=lr, momentum=0.9)
        
        for step in range(20):
            out = w[0] * x[0] + b[0]
            target = Value(10.0)
            loss = (out - target) ** 2
            
            loss.backward()
            opt.step()
            
            if step % 5 == 0:
                print(f"  Step {step:2d}: loss={loss.data:.6f}, w={w[0].data:.4f}")
        
        return w[0].data
    
    print("\nTarget: w*x + b = 10.0, x = 2.0")
    print("Expected w ~= 5.0, b ~= 0.0")
    
    run_optimizer(SGD, "SGD", lr=0.1)
    run_optimizer(Adam, "Adam", lr=0.1)
    run_optimizer(AdamW, "AdamW", lr=0.1)


def demo_mlp():
    """Demo MLP dengan XOR problem"""
    print("\n" + "=" * 60)
    print("DEMO: MLP untuk XOR Problem")
    print("=" * 60)
    
    X = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    y = [0, 1, 1, 0]
    y_values = [Value(t) for t in y]
    
    def get_pred_value(pred):
        if isinstance(pred, list):
            return pred[0]
        return pred
    
    model = MLP(2, [8, 4, 1])
    optimizer = Adam(model.parameters(), lr=0.5)
    
    for epoch in range(500):
        preds = [model(x) for x in X]
        loss = sum((get_pred_value(p) - t) ** 2 for p, t in zip(preds, y_values)) / len(X)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.data:.6f}")
    
    print("\nPrediksi setelah training:")
    for i, x in enumerate(X):
        pred = model(x)
        pred_val = get_pred_value(pred)
        print(f"  {int(x[0].data)} XOR {int(x[1].data)} = {pred_val.data:.4f} (target: {y[i]})")


def demo_autograd():
    """Demo fitur AutoGrad"""
    print("\n" + "=" * 60)
    print("DEMO: AutoGrad (Value Class)")
    print("=" * 60)
    
    print("\n--- Operasi Matematika ---")
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = a * b + Value(10, label='c')
    c.label = 'output'
    c.backward()
    
    print(f"  a = {a.data}, b = {b.data}")
    print(f"  c = a * b + 10 = {c.data}")
    print(f"  dc/da = {a.grad} (should be -3.0)")
    print(f"  dc/db = {b.grad} (should be 2.0)")
    
    print("\n--- Aktivasi ---")
    x = Value(1.5)
    print(f"  x = {x.data}")
    
    for name, act_fn in [('ReLU', lambda v: v.relu()),
                         ('Tanh', lambda v: v.tanh()),
                         ('Sigmoid', lambda v: v.sigmoid())]:
        out = act_fn(x)
        out.backward()
        print(f"  {name}({x.data:.2f}) = {out.data:.4f}, grad = {x.grad:.4f}")


if __name__ == "__main__":
    demo_comparison()
    demo_mlp()
    demo_autograd()
    
    print("\n" + "=" * 60)
    print("Semua demo selesai!")
    print("=" * 60)
