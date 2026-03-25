# NOTES

## [Homework 1](https://github.com/ChristFarrell/_ml/blob/master/Homework/Homework%2001%20050326/Climb.py)

This homework was getting helped by AI for help understanding.<br>
Link for AI : https://gemini.google.com/share/63b6aadd2bfbb <br>

This project works by using climbing techniques to find the fastest final distance. At first we started by making 10 random seed from 0 until 100 in coordinat of x and y. The program also will calculate the distance for each seed.
```python
random.seed(42)  # Ensures consistent coordinates across runs
city_locations = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(10)}

def calculate_distance(c1, c2):
    (x1, y1), (x2, y2) = city_locations[c1], city_locations[c2]
    # Standard Euclidean distance formula
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
```

After that program will trying the some method. The first method is using Hill Climbing. Hill Climbing Essentially, the program wants to find the highest "peak." The second method is neighbor (2-opt). The program selects two sections of road at random, then rotates their order so that the "crossed" roads become "straight"
```python
def height(self):
        """Condition: Total Distance * -1 (Higher height = shorter distance)"""
        total = 0
        for i in range(len(self.route) - 1):
            total += calculate_distance(self.route[i], self.route[i+1])
        total += calculate_distance(self.route[-1], self.route[0])  # Return to start
        return -total

    def neighbor(self):
        """Condition: 2-opt swap (a,b)(c,d) -> (a,d)(b,c)"""
        new_route = list(self.route)
        # Select two random indices
        i, j = sorted(random.sample(range(len(new_route)), 2))
        # Reverse the segment between i and j
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return TSPSolution(new_route)
```

After that, the engine started when program by starting with a random route, for example, from house 1 to 10. The program will then:
- Continue swapping the order of the houses.
- Check the distance, if the swapping results in a shorter distance.
- Give up: If the program has tried swapping up to 500 times but no other route options are found.

## [Homework 2](https://github.com/ChristFarrell/_ml/blob/master/Homework/Homework%2002%20120326/Manual%20Calculation%20of%20The%20Inverse%20Transit%20Algorithm.jpeg)

Problem 1: $f(x, y, z) = (x \cdot y) + z$<br>
In this computational graph, we first perform a Forward Pass where the inputs $x=1$ and $y=2$ are multiplied to produce an intermediate value of $2$, which is then added to $z=3$ to reach a final output of $5$.
| Part | Operation / Variable | Chain Rule Formula | Final Gradient |
| :--- | :--- | :--- | :--- |
| **Addition** | $P + z$ | $\frac{\partial f}{\partial P}$ and $\frac{\partial f}{\partial z}$ | **1** |
| **z** | Input $z$ | $\frac{\partial f}{\partial z}$ | **1** |
| **Multiplication** | $x * y$ | $\frac{\partial f}{\partial P}$ | **1** |
| **x** | Input $x$ | $\frac{\partial f}{\partial P} \cdot y$ | **2** |
| **y** | Input $y$ | $\frac{\partial f}{\partial P} \cdot x$ | **1** |

Problem 2: $f(x, y, z, t) = ((x \cdot y) + z) \cdot t$<br>
In the Forward Pass, the product of $x(1)$ and $y(2)$ results in $2$, which is added to $z(3)$ to get a sub-total of $5$, and finally multiplied by $t(4)$ to yield a total of $20$.
| Part | Operation / Variable | Chain Rule Formula | Final Gradient |
| :--- | :--- | :--- | :--- |
| **Final Multi** | $Q * t$ | $\frac{\partial f}{\partial Q}$ and $\frac{\partial f}{\partial t}$ | $Q \rightarrow 4, t \rightarrow 5$ |
| **t** | Input $t$ | $\frac{\partial f}{\partial t}$ | **5** |
| **Addition** | $P + z$ | $\frac{\partial f}{\partial Q} \cdot 1$ | **4** |
| **z** | Input $z$ | $\frac{\partial f}{\partial z}$ | **4** |
| **First Multi** | $x * y$ | $\frac{\partial f}{\partial P}$ | **4** |
| **x** | Input $x$ | $\frac{\partial f}{\partial P} \cdot y$ | **8** |
| **y** | Input $y$ | $\frac{\partial f}{\partial P} \cdot x$ | **4** |

## [Homework 3](https://github.com/ChristFarrell/_ml/tree/master/Homework/Homework%2003%20190326)

This homework was getting helped by Opencode AI for help understanding.<br>

This project demonstrates fundamental concepts in Neural Networks including:
- **AutoGrad Engine** - Automatic differentiation
- **8 Optimizers** - SGD, Momentum, Nesterov, RMSProp, Adagrad, AdaDelta, Adam, AdamW
- **Gradient Descent** - Training with live visualization

Value Class (AutoGrad)
The `Value` class is the fundamental unit that tracks:
- **data**: The actual value
- **grad**: The gradient (derivative)
- **_prev**: Children nodes (computation graph)
- **_backward**: Function to compute gradients

Forward & Backward Pass
```python
w = Value(0.5)      # weight
x = Value(2.0)       # input
b = Value(0.0)       # bias

logit = w * x + b   # forward: 0.5 * 2.0 + 0.0 = 1.0
```

**Backward Pass**: Compute gradients via chain rule
```python
loss = (logit - 10) ** 2  # MSE: (1.0 - 10)^2 = 81
loss.backward()           # compute d(loss)/dw, d(loss)/db
```

Optimizers, All optimizers update parameters using the general form:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta} \mathcal{L}$$

where $\eta$ is the learning rate and $\nabla_{\theta} \mathcal{L}$ is the gradient.

| # | Optimizer | Update Formula | Characteristics |
|:-:|:---------:|:--------------|:---------------|
| 1 | **SGD** (Stochastic Gradient Descent) | $\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}$ | Simplest optimizer. Updates based on raw gradient. Can get stuck in local minima. |
| 2 | **SGD + Momentum** | $v_t = \beta v_{t-1} + (1-\beta)\nabla_{\theta}\mathcal{L}$ | Adds velocity to escape local minima. $\beta$ typically = 0.9. |
| 3 | **Nesterov Accelerated Gradient** | $v_t = \beta v_{t-1} + \nabla_{\theta}\mathcal{L}(\theta_t - \beta v_{t-1})$ | "Look-ahead" gradient. More accurate than standard momentum. |
| 4 | **RMSProp** | $E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$<br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$ | Adaptive learning rate per parameter. Good for non-stationary problems. |
| 5 | **Adagrad** | $G_t = G_{t-1} + g_t \odot g_t$<br>$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$ | Accumulates squared gradients. Automatically adjusts learning rate. |
| 6 | **AdaDelta** | $E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho)g_t^2$<br>$\Delta\theta_t = -\frac{\sqrt{E[\Delta\theta^2]_{t-1}+\epsilon}}{\sqrt{E[g^2]_t+\epsilon}}g_t$ | No learning rate hyperparameter. Uses running average of gradient changes. |
| 7 | **Adam** (Adaptive Moment Estimation) | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$<br>$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$<br>$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \hat{v}_t = \frac{v_t}{1-\beta_2^t}$<br>$\theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ | Combines momentum + RMSProp. Most popular optimizer. Default: $\beta_1=0.9$, $\beta_2=0.999$. |
| 8 | **AdamW** (Adam with Weight Decay) | $g_t = \nabla_{\theta}\mathcal{L} + \lambda\theta_t$<br>Then apply Adam update with $g_t$ | Proper weight decay implementation. Better regularization than L2. |

Loss Functions
| Loss Function | Formula | Use Case | Characteristics |
|:-------------:|:--------:|:--------:|:---------------|
| **MSE** | $\displaystyle \mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ | Regression | Penalizes large errors heavily (quadratic penalty). Sensitive to outliers. |
| **Cross-Entropy** | $\displaystyle \mathcal{L}_{\text{CE}} = -\sum_{i} y_i \log(\hat{y}_i)$ | Multi-class Classification | Probabilistic outputs. Measures divergence between predicted and true distributions. |
| **Binary Cross-Entropy** | $\displaystyle \mathcal{L}_{\text{BCE}} = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]$ | Binary Classification | Works with sigmoid activation. Ideal for binary classification problems. |

Activation Functions
| Activation | Formula | Derivative | Range | Characteristics |
|:----------:|:-------:|:----------:|:-----:|:---------------|
| **ReLU** | $f(x) = \max(0, x)$ | $f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | $[0, \infty)$ | Most popular. Fast computation. Suffers from "dying ReLU" problem. |
| **Tanh** | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $f'(x) = 1 - f(x)^2$ | $(-1, 1)$ | Zero-centered output. Better for hidden layers than sigmoid. |
| **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | $f'(x) = f(x)(1 - f(x))$ | $(0, 1)$ | Used for binary classification output. Prone to vanishing gradients. |

Training Example
```python
# Target: w * x + b = 10.0, where x = 2.0
# Expected: w ≈ 5.0, b ≈ 0.0

x = Value(2.0)      # input
w = Value(0.5)      # weight (initial)
b = Value(0.0)      # bias (initial)

optimizer = Adam([w, b], lr=0.1)

for epoch in range(100):
    logit = w * x + b           # forward
    loss = (logit - 10) ** 2    # MSE
    loss.backward()              # backward
    optimizer.step()            # update
```

Parameter Reference
| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Learning Rate | $\eta$ | 0.01 - 0.1 | Step size for parameter update |
| Beta 1 | $\beta_1$ | 0.9 | Decay rate for first moment (momentum) |
| Beta 2 | $\beta_2$ | 0.999 | Decay rate for second moment (RMS) |
| Momentum | $\beta$ | 0.9 | Velocity in SGD momentum |
| Epsilon | $\epsilon$ | $10^{-8}$ | Numerical stability constant |
| Weight Decay | $\lambda$ | 0.01 | L2 regularization strength |
| Rho | $\rho$ | 0.95 | Decay rate for AdaDelta |

## References
- [Adam Paper](https://arxiv.org/abs/1412.6980)
- [Optimization for Deep Learning](https://ruder.io/optimizing-gradient-descent/)
