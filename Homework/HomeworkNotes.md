# NOTES

## [Homework 1](https://github.com/ChristFarrell/_ml/blob/master/Homework/Homework%201%20050326/Climb.py)

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

## [Homework 2](https://github.com/ChristFarrell/_ml/blob/master/Homework/Homework%202%20120326/Manual%20Calculation%20of%20The%20Inverse%20Transit%20Algorithm.jpeg)

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

## [Homework 3](https://github.com/ChristFarrell/_ml/tree/master/Homework/Homework%203%20190326)

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
| 2 | **SGD + Momentum** | $v_t = \beta v_{t-1} + (1-\beta)\nabla_{\theta}\mathcal{L}$ | Adds velocity to escape local minima. $\beta$ typically = 0.9 |
| 3 | **Nesterov Accelerated Gradient** | $v_t = \beta v_{t-1} + \nabla_{\theta}\mathcal{L}(\theta_t - \beta v_{t-1})$ | "Look-ahead" gradient. More accurate than standard momentum |
| 4 | **RMSProp** | $E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)g_t^2$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$ | Adaptive learning rate per parameter. Good for non-stationary problems. |
| 5 | **Adagrad** | $G_t = G_{t-1} + g_t^2$, $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$ | Accumulates squared gradients. Automatically adjusts learning rate. |
| 6 | **AdaDelta** | $E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho)g_t^2$, $\Delta\theta_t = -\frac{\sqrt{E[\Delta\theta^2]_{t-1}+\epsilon}}{\sqrt{E[g^2]_t+\epsilon}}g_t$ | No learning rate hyperparameter. Uses running average of gradient changes. |
| 7 | **Adam** | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$, $\theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$ | Combines momentum + RMSProp. Default: $\beta_1=0.9, \beta_2=0.999$ |
| 8 | **AdamW** | $g_t = \nabla_{\theta}\mathcal{L} + \lambda\theta_t$ then apply Adam | Proper weight decay. Better regularization than L2. |

Loss Functions
| Loss Function | Formula | Use Case | Characteristics |
|:-------------:|:--------:|:--------:|:---------------|
| **MSE** | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Regression | Penalizes large errors heavily. Sensitive to outliers. |
| **Cross-Entropy** | $-\sum y_i \log(\hat{y}_i)$ | Multi-class Classification | Probabilistic outputs. Measures divergence between distributions. |
| **Binary Cross-Entropy** | $-(y \log(\hat{y}) + (1-y)\log(1-\hat{y}))$ | Binary Classification | Works with sigmoid. Ideal for binary problems. |

Activation Functions
| Activation | Formula | Derivative | Range | Characteristics |
|:----------:|:-------:|:----------:|:-----:|:---------------|
| **ReLU** | $\max(0, x)$ | $1$ if $x>0$, else $0$ | $[0, \infty)$ | Fast computation. Suffers from "dying ReLU". |
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

References
- [Adam Paper](https://arxiv.org/abs/1412.6980)
- [Optimization for Deep Learning](https://ruder.io/optimizing-gradient-descent/)


## [Homework 4](https://github.com/ChristFarrell/_ml/tree/master/Homework/Homework%204%20260326)

This homework was getting helped by Opencode AI for help understanding.<br>
More explanation : https://github.com/ChristFarrell/_ml/blob/master/Homework/Homework%204%20260326/README.md<Br>
Based on the concepts from Andrej Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).<br>

| Component | Description |
|-----------|-------------|
| **Value class** | Scalar autograd engine for backpropagation |
| **Tokenizer** | Character-level: each letter gets an integer id |
| **GPT architecture** | Token + position embeddings, multi-head attention, MLP, residuals, RMSNorm |
| **Optimizer** | Adam with linear learning rate decay |
| **Training** | Cross-entropy loss on next-token prediction |
| **Inference** | Temperature-controlled sampling to generate new names |

On the training, it would shown like this.
```
num docs: 32033
vocab size: 27
num params: 1296

--- training for 500 steps ---

step    1 / 500 | loss 3.3686
...
step  500 / 500 | loss 2.3908
```

After training, the model generates new names:
```
--- inference (temperature=0.5) ---

  ienis
  saleri
  jala
  talan
  janaen
  ...
```

```python
n_embd = 8       # increase for more capacity (e.g. 16, 32)
n_head = 2       # must divide n_embd evenly
n_layer = 1      # increase for deeper model
block_size = 12  # max sequence length
num_steps = 200  # train longer for better results
temperature = 0.5  # lower = conservative, higher = creative
```

| Config | Params | Steps | Quality |
|--------|--------|-------|---------|
| `n_embd=8, n_head=2, n_layer=1` | 1,296 | 200 | Basic |
| `n_embd=16, n_head=4, n_layer=1` | 4,192 | 500 | Better |
| `n_embd=16, n_head=4, n_layer=2` | ~8,000 | 1000 | Good |

The concept of work for this project can be shown like this.
```
Input:  "emma"
Tokens: [BOS, e, m, m, a, BOS]

For each position, the model predicts the next token:

  BOS → e
  e   → m
  m   → m
  m   → a
  a   → BOS

Loss = average of -log(probability assigned to correct token)
```
The model learns statistical patterns in names (consonant-vowel structure, common beginnings/endings) and generates new plausible names by sampling from its learned distribution.

## [Homework 5](https://github.com/ChristFarrell/_ml/tree/master/Homework/Homework%205%20020426/v2-agent-xml)

This homework was getting helped by Opencode AI for help understanding.<br>

On this homework, we asked to modify the agent0.py to update security controls: It asked
1. Files outside the program's folder cannot be directly accessed (internal files are allowed).
2. If an agent0.py attempts to access an external file, it must first block the access and request permission before allowing access.

The update python shown in the feature:
- SCRIPT_DIR & APPROVED_PATHS	    [17-18]
- is_path_within_allowed()	        [91-114]
- check_and_approve()	            [116-138]
- Security check in main loop	    [201-205]
- Print security scope on startup	[160]

The project flow was progress as below:
1. Startup
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) captures the program's folder path as the security boundary.

2. Command interception
    Before subprocess.run() executes, check_and_approve(cmd) is called.

3. Path validation
    check_and_approve() calls is_path_within_allowed(cmd) to extract file paths from the shell command and compare them against SCRIPT_DIR.

4. Decision flow:
    - If all paths are within SCRIPT_DIR → return True → command executes
    - If external paths found → print warning → ask user: yes/no/always
        - yes → return True → command executes
        - no → return False → command blocked
        - always → store path in APPROVED_PATHS → future commands to same path auto-approved

## [Homework 6](https://github.com/ChristFarrell/_ml/tree/master/Homework/Homework%206%20230426)

This homework was getting helped by Claude for help understanding.<br>
Link for Claude: https://claude.ai/share/61787149-4a89-457e-bae6-3e971058ba15<br>

On this code, we asked to made the mini-gpt without using the transformer. It means, we made the chat-bot with manual way without using any implementation chat AI. The chat work by 2 category, rule-based or generated. This is showing of flow chart of machine
```
You type something
        ↓
Does it match a rule? ("hello", "who are you", "what is ai", etc.)
        ↓
    YES → return the hand-written answer immediately
          (no temperature, no top-k, no model involved at all)
        ↓
    NO  → pass your input to the language model
              ↓
          Take last 2 words of your input
              ↓
          LogReg outputs probability for every word
              ↓
          Apply top-k  → cut out low-probability words
              ↓
          Apply temperature → reshape the distribution
              ↓
          Sample a word → append it → repeat
```

1. Rule Based<br>
    On rule based, we already put some dictionary inside of our chat-bot, so if the user's word have match with the bot dictionary, bot can taking the dictionary word to be printed out. for example
    ```
    "hello":              ["Hello! How can I help you today?", "Hi there! Nice to meet you!"],
    "hi":                 ["Hi! How are you?", "Hello!"],
    "what is ai":         ["AI stands for Artificial Intelligence — machines that simulate human thinking
                            ", "AI is the field of making computers learn, reason, and adapt."],
    "what is python":     ["Python is a popular programming language great for data science and AI.",
                            "Python is known for its clear syntax and rich ML ecosystem."],
    ```

    Here is the example output of using the dictionary chat-bot:
    ```
    You: hi
    Mini GPT: Hello!
         [mode: rule-based]  [temp: —  top-k: —]  [UNK: hi]

    You: how are you
    Mini GPT: Great! How can I help?
         [mode: rule-based]  [temp: —  top-k: —]  [UNK: you]

    You: what is ai
    Mini GPT: AI is the field of making computers learn, reason, and adapt.
         [mode: rule-based]  [temp: —  top-k: —]  [UNK: what, ai]

    You: what is ml
    Mini GPT: Machine Learning is a branch of AI where systems improve from experience.
         [mode: rule-based]  [temp: —  top-k: —]  [UNK: what, ml]
    ```

2. Generated<br>
    For the word that was out of context or not included in dictionary, the training can be using so bot can predict the answer of chat. The tokens that we use was <EOS> (end of sentence) and <UNK> (unknown word). For example, from the sentence "machine learning allows computers to learn":
    | Context (input) | Next word (target) |
    | :--- | :--- |
    | `<EOS> <EOS>` | machine |
    | `<EOS> machine` | learning |
    | `machine learning` | allows |
    | `learning allows` | computers |
    | `allows computers` | to |
    | `computers to` | learn |
    | `to learn` | `<EOS>` |

    Since machine work by mathemathic, it translates the word to context of number. If the machine have detect word that was match and already saved to their memory, machine will give One-Hot Vector or x matrix. As long the context from x matrix, the y vector was a target word. The machine detects it by see the context. For example:
    ```
    <EOS>, machine, learning, allows
    Machine translate: 0 1 0 0
    
    Context (Input X): <EOS> machine
    Next Word (Target): learning
    y value: Because learning in Index 2, so y value = 2
    ```

    The context of this x and y, is that:
    - The AI ​​looks at Input X (the numbers 0 and 1).
    - The AI ​​tries to guess. For example, it guesses 3 (allows).
    - The system checks Target y. It turns out the correct answer is 2 (learning).
    - Because the guess was wrong (3 ≠ 2), the AI ​​will improve (update its weights) so that the next time its guess is closer to 2.

    On process generating:
    - Takes the last 2 words of your input as context
    - Asks LogReg: "what's the probability of every word in the vocabulary coming next?"
    - Samples from that distribution
    - Appends that word to the sentence, then uses the new last 2 words as the next context
    - Repeats until it hits <EOS> or reaches the word limit

    At the end, here is the example output of using the generated chat-bot
    ```
    You: machine learning allows computers to learn
    Mini GPT: to learn communicate
         [mode: generated]  [temp: 0.9  top-k: 10]  [all words known]

    You: banana is fruit
    Mini GPT: is fruit offers
         [mode: generated]  [temp: 0.9  top-k: 10]  [UNK: banana, fruit]
    ```