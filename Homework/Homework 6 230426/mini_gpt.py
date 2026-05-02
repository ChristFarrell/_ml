"""
Mini GPT - Non-Transformer Language Model Chatbot
==================================================
Uses sklearn LogisticRegression to learn word probabilities
from context (n-gram style). No transformers, no neural networks.

Architecture:
  - N-gram context (default: order=2) → one-hot features
  - LogisticRegression predicts P(next_word | context)
  - Sampling with temperature & top-k for generation
"""

import random
import numpy as np
import pickle
import os
from collections import defaultdict


# ─────────────────────────────────────────────
#  Word Classifier  (n-gram + LogReg language model)
# ─────────────────────────────────────────────

class WordClassifier:
    """
    A simple n-gram language model backed by scikit-learn
    LogisticRegression. The model predicts the next word
    given the previous `order` words, using one-hot encoding.
    """

    SPECIAL_EOS = "<EOS>"
    SPECIAL_UNK = "<UNK>"

    def __init__(self, order: int = 2):
        self.order = order           # n-gram window size
        self.word_to_idx: dict = {}
        self.idx_to_word: dict = {}
        self.vocab: set = set()
        self.model = None

    # ── Training ──────────────────────────────

    def train(self, sentences: list[str], max_iter: int = 500) -> None:
        """
        Build vocabulary and train the LogisticRegression model.

        Each training sample is:
          X  = one-hot(word[i-order]) + ... + one-hot(word[i-1])
          y  = index of word[i]
        """
        # Build vocabulary
        for sentence in sentences:
            self.vocab.update(sentence.lower().split())
        self.vocab.add(self.SPECIAL_EOS)
        self.vocab.add(self.SPECIAL_UNK)

        self.word_to_idx = {w: i for i, w in enumerate(sorted(self.vocab))}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}

        vocab_size = len(self.word_to_idx)

        # Build feature matrix
        X, y = [], []
        for sentence in sentences:
            words = sentence.lower().split()
            # Pad start with EOS tokens so every word gets a context
            padded = [self.SPECIAL_EOS] * self.order + words + [self.SPECIAL_EOS]

            for i in range(len(words) + 1):   # +1 to predict the final EOS
                context = padded[i : i + self.order]
                target  = padded[i + self.order]
                X.append(self._words_to_features(context, vocab_size))
                y.append(self.word_to_idx[target])

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
        self.model.fit(X, y)
        print(f"[Trained] vocab={vocab_size}  samples={len(X)}  order={self.order}")

    # ── Feature encoding ──────────────────────

    def _words_to_features(self, words: list[str], vocab_size: int | None = None) -> list[float]:
        """Concatenate one-hot vectors for each word in the context window."""
        if vocab_size is None:
            vocab_size = len(self.word_to_idx)
        features = []
        for w in words:
            vec = [0.0] * vocab_size
            idx = self.word_to_idx.get(w, self.word_to_idx.get(self.SPECIAL_UNK, 0))
            vec[idx] = 1.0
            features.extend(vec)
        return features

    def _get_features(self, context_words: list[str]) -> np.ndarray:
        """Prepare context window features as a (1, D) numpy array."""
        words = list(context_words)
        # Pad if shorter than order
        while len(words) < self.order:
            words.insert(0, self.SPECIAL_UNK)
        words = words[-self.order:]   # keep only last `order` words
        return np.array([self._words_to_features(words)], dtype=np.float32)

    # ── Prediction ────────────────────────────

    def predict_probs(
        self,
        context: str,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> np.ndarray:
        """
        Return probability distribution over the vocabulary.

        Args:
            context:     Space-separated string of preceding words.
            temperature: >1 = more random, <1 = more peaked. 1.0 = no change.
            top_k:       If >0, zero out all but the top-k tokens before re-normalising.
        """
        words = context.lower().split()[-self.order:]
        features = self._get_features(words)
        probs = self.model.predict_proba(features)[0].astype(np.float64)

        # Temperature scaling (applied in log space for numerical stability)
        if temperature != 1.0:
            log_probs = np.log(probs + 1e-12) / temperature
            probs = np.exp(log_probs - log_probs.max())
            probs /= probs.sum()

        # Top-k filtering
        if top_k > 0:
            top_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_indices] = 1.0
            probs = probs * mask
            probs /= probs.sum()

        return probs

    def predict_next(self, context: str, temperature: float = 1.0, top_k: int = 0) -> str:
        """Sample one word from the predicted distribution."""
        probs = self.predict_probs(context, temperature, top_k)
        idx = np.random.choice(len(probs), p=probs)
        return self.idx_to_word[idx]

    def generate(
        self,
        seed: str,
        max_words: int = 25,
        temperature: float = 1.0,
        top_k: int = 10,
    ) -> str:
        """
        Auto-regressively generate text starting from `seed`.
        Generation stops at <EOS> or when max_words is reached.
        """
        words = seed.lower().split()[-self.order:]

        for _ in range(max_words):
            next_word = self.predict_next(" ".join(words), temperature, top_k)
            if next_word == self.SPECIAL_EOS:
                break
            words.append(next_word)

        # Strip leading padding tokens
        output = [w for w in words if w not in (self.SPECIAL_EOS, self.SPECIAL_UNK)]
        return " ".join(output)

    # ── Persistence ───────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "word_to_idx": self.word_to_idx,
                "idx_to_word": self.idx_to_word,
                "model":       self.model,
                "vocab":       self.vocab,
                "order":       self.order,
            }, f)
        print(f"[Saved] model → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.word_to_idx = data["word_to_idx"]
        self.idx_to_word = data["idx_to_word"]
        self.model       = data["model"]
        self.vocab       = data["vocab"]
        self.order       = data["order"]
        print(f"[Loaded] model ← {path}  (vocab={len(self.vocab)})")


# ─────────────────────────────────────────────
#  Chatbot
# ─────────────────────────────────────────────

class MiniGPTChatBot:
    """
    Rule-based responses for common greetings / meta questions,
    falling back to the language model for everything else.
    """

    RULES: dict[str, list[str]] = {
        "hello":              ["Hello! How can I help you today?", "Hi there! Nice to meet you!"],
        "hi":                 ["Hi! How are you?", "Hello!"],
        "hey":                ["Hey! What's up?", "Hello there!"],
        "who are you":        ["I'm Mini GPT, a language model using sklearn — no transformers!",
                               "I'm a chatbot built with LogisticRegression."],
        "what is your name":  ["My name is Mini GPT.", "You can call me Mini GPT."],
        "how are you":        ["I'm doing well, thanks! How about you?", "Great! How can I help?"],
        "how do you work":    ["I use LogisticRegression to predict the next word from the previous two words.",
                               "I model language as an n-gram chain — no neural nets needed!"],
        "what are you":       ["I'm a language model without transformers.",
                               "I'm built with scikit-learn LogisticRegression."],
        "bye":                ["Goodbye! Nice chatting with you!", "See you later!"],
        "goodbye":            ["Goodbye! Have a great day!", "Bye! Take care!"],
        "thank you":          ["You're welcome!", "Happy to help!"],
        "thanks":             ["You're welcome!", "No problem at all!"],
        "help":               ["I can chat with you! Try asking who I am, or ask about AI and Python.",
                               "Ask me anything — I'll do my best to answer."],
        "what is ai":         ["AI stands for Artificial Intelligence — machines that simulate human thinking.",
                               "AI is the field of making computers learn, reason, and adapt."],
        "what is python":     ["Python is a popular programming language great for data science and AI.",
                               "Python is known for its clear syntax and rich ML ecosystem."],
        "what is ml":         ["ML stands for Machine Learning — algorithms that learn from data.",
                               "Machine Learning is a branch of AI where systems improve from experience."],
        "what is deep learning": ["Deep learning uses multi-layer neural networks to learn features from raw data.",
                                  "Deep learning is a subset of ML inspired by the human brain."],
        "what is nlp":        ["NLP is Natural Language Processing — teaching machines to understand text.",
                               "NLP lets computers read, understand, and generate human language."],
    }

    def __init__(self, model: WordClassifier):
        self.model = model

    def detect_unknown(self, user_input: str) -> list[str]:
        """Return words in user_input that are not in the model vocabulary."""
        words = user_input.lower().split()
        return [w for w in words if w not in self.model.vocab]

    def respond(
        self,
        user_input: str,
        temperature: float = 0.9,
        top_k: int = 10,
    ) -> dict:
        """
        Process user input and return a result dict with:
          - reply     : the bot's response string
          - mode      : "rule" or "generated"
          - temp      : temperature used (None if rule-based)
          - top_k     : top-k used (None if rule-based)
          - unknowns  : list of words not found in vocabulary
        """
        text = user_input.lower().strip()
        unknowns = self.detect_unknown(text)

        # Rule-based matching (whole-word, longest key wins on tie)
        import re
        matched_key = None
        for key in sorted(self.RULES, key=len, reverse=True):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, text):
                matched_key = key
                break

        if matched_key:
            return {
                "reply":    random.choice(self.RULES[matched_key]),
                "mode":     "rule",
                "temp":     None,
                "top_k":    None,
                "unknowns": unknowns,
            }

        # Fall back to generative model
        response = self.model.generate(text, max_words=20, temperature=temperature, top_k=top_k)

        # If generation is too short, retry with higher temperature
        if len(response.split()) < 3:
            response = self.model.generate(text, max_words=20, temperature=1.4, top_k=0)

        return {
            "reply":    response if response else "Hmm, I'm not sure how to respond to that!",
            "mode":     "generated",
            "temp":     temperature,
            "top_k":    top_k,
            "unknowns": unknowns,
        }


# ─────────────────────────────────────────────
#  Training data loader
# ─────────────────────────────────────────────

def load_training_data(path: str) -> list[str]:
    """Read a text file and return non-empty lines as training sentences."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"[Data] loaded {len(lines)} sentences from {path}")
    return lines


DEFAULT_TRAINING_SENTENCES: list[str] = [
    # AI & ML
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "Machine learning allows computers to learn from data without being explicitly programmed.",
    "Deep learning uses neural networks with many layers to learn complex patterns.",
    "Natural language processing helps computers understand and generate human text.",
    "Supervised learning trains models on labeled examples.",
    "Unsupervised learning discovers hidden patterns in unlabeled data.",
    "Reinforcement learning trains agents through rewards and penalties.",
    "Logistic regression predicts the probability of a class label.",
    "A language model predicts the probability of the next word given context.",
    "Transformers use attention mechanisms to process sequences in parallel.",
    "N-gram models predict words based on a fixed window of previous words.",
    "Overfitting occurs when a model memorises training data but fails to generalise.",
    "Regularisation techniques like L2 penalty help reduce overfitting.",
    "Cross-validation estimates how well a model generalises to unseen data.",
    "Feature engineering transforms raw data into useful representations.",
    "One-hot encoding represents categorical variables as binary vectors.",
    "Vocabulary is the set of unique words that a language model knows.",
    "Temperature controls how random or deterministic text generation is.",
    "Top-k sampling limits generation to the k most probable next words.",
    # Python & Programming
    "Python is a high-level programming language popular in data science.",
    "NumPy provides fast numerical operations on multi-dimensional arrays.",
    "Scikit-learn offers simple tools for predictive data analysis.",
    "A function is a reusable block of code that performs a specific task.",
    "Object-oriented programming organises code into classes and objects.",
    "Data structures like lists and dictionaries are fundamental in Python.",
    "APIs allow different software systems to communicate with each other.",
    "Version control with Git helps teams collaborate on code.",
    # General knowledge
    "The internet connects billions of devices and people around the world.",
    "Open source software allows anyone to view and modify the source code.",
    "Robots are machines programmed to perform tasks automatically.",
    "Computers process information using binary digits called bits.",
    "Algorithms are step-by-step procedures for solving problems.",
    "Data science combines statistics, programming, and domain expertise.",
    "The cloud refers to servers and services accessed over the internet.",
    "Cybersecurity protects computer systems from theft and damage.",
    "Automation reduces repetitive manual work through software and machines.",
    "Big data refers to extremely large datasets that require special tools.",
]


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Mini GPT  —  sklearn Word Classifier (no transformers)")
    print("=" * 60)
    print(
        "\nHow it works:\n"
        "  1. Sentence → sliding window of 2 words (n-gram context)\n"
        "  2. Context encoded as one-hot feature vector\n"
        "  3. LogisticRegression learns P(next word | context)\n"
        "  4. At inference: sample from distribution with temperature\n"
        "  5. Repeat until <EOS> or max_words reached\n"
    )

    MODEL_PATH = "mini_gpt.pkl"
    lm = WordClassifier(order=2)

    if os.path.exists(MODEL_PATH):
        lm.load(MODEL_PATH)
        print("[Info] Loaded cached model. Delete mini_gpt.pkl to retrain.\n")
    else:
        print("[Info] Training on built-in dataset…")
        lm.train(DEFAULT_TRAINING_SENTENCES, max_iter=1000)
        lm.save(MODEL_PATH)

    bot = MiniGPTChatBot(lm)

    # Default generation settings (user can change these live)
    temperature = 0.9
    top_k       = 10

    print("\nChat with Mini GPT  (type 'quit' or 'exit' to stop)")
    print("Commands: !temp <value>  /  !topk <value>  to adjust settings")
    print("-" * 60)
    print(f"[Settings] temperature={temperature}  top_k={top_k}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Mini GPT: Goodbye! It was nice chatting with you.")
            break

        # Live setting adjustments
        if user_input.startswith("!temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"[Settings] temperature updated → {temperature}\n")
            except ValueError:
                print("[Error] Usage: !temp <float>  e.g. !temp 1.2\n")
            continue

        if user_input.startswith("!topk "):
            try:
                top_k = int(user_input.split()[1])
                print(f"[Settings] top_k updated → {top_k}\n")
            except ValueError:
                print("[Error] Usage: !topk <int>  e.g. !topk 5\n")
            continue

        result = bot.respond(user_input, temperature=temperature, top_k=top_k)

        print(f"Mini GPT: {result['reply']}")

        # ── Score line ──
        if result["mode"] == "rule":
            mode_str  = "[mode: rule-based]"
            score_str = "[temp: —  top-k: —]"
        else:
            mode_str  = "[mode: generated]"
            score_str = f"[temp: {result['temp']}  top-k: {result['top_k']}]"

        if result["unknowns"]:
            unk_str = f"[UNK: {', '.join(result['unknowns'])}]"
        else:
            unk_str = "[all words known]"

        print(f"         {mode_str}  {score_str}  {unk_str}\n")


if __name__ == "__main__":
    main()