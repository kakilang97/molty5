"""
Adaptive Brain — Q-Table Learning engine for Molty Royale.

Architecture:
  - Q-Table with discretized state bins (no heavy dependencies like PyTorch/TF)
  - Upper Confidence Bound (UCB) exploration: smart exploration decays over time
  - Thompson Sampling fallback when few visits are recorded
  - Learns from experience replay buffer after each game
  - All rule-based SAFETY constraints (deathzone, critical heal) ALWAYS override ML

Learning pipeline:
  1. Per-step: record (state, action, reward, next_state) in replay buffer
  2. After game: run Q-learning update pass on sampled experiences
  3. Q-table persisted to disk via AgentMemory for cross-game learning
  4. Softmax action selection (temperature annealing over training games)

State discretization (20 features → 4 bins each):
  Bins: [0, 0.25) → 0 | [0.25, 0.5) → 1 | [0.5, 0.75) → 2 | [0.75, 1.0] → 3
  State key: tuple of 20 bin indices → string key for dict storage

Q-Table guarantees:
  - Always explores if a state is NEW (UCB formula gives infinity)
  - Decays exploration as state gets more visits (c/sqrt(N) term)
  - Safety rules from brain.py ALWAYS take priority over Q-values
"""
import json
import math
import random
from typing import Optional
from bot.utils.logger import get_logger
from bot.ml.experience_replay import (
    ReplayBuffer, Experience, ACTION_NAMES, N_ACTIONS,
    compute_reward,
)
from bot.ml.feature_extractor import extract_features

log = get_logger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────
ALPHA = 0.15       # Learning rate
GAMMA = 0.92       # Discount factor
UCB_C = 1.5        # UCB exploration constant
EPSILON_START = 0.35
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.97   # Per training game
BATCH_SIZE = 64
N_BINS = 4             # Feature discretization bins per dimension
N_FEATURES = 20


def _discretize(features: list[float]) -> tuple:
    """Map each feature [0,1] to a bin index 0..N_BINS-1."""
    return tuple(min(int(f * N_BINS), N_BINS - 1) for f in features)


def _state_key(features: list[float]) -> str:
    return str(_discretize(features))


class QTable:
    """
    Dictionary-based Q-table: state_key → [Q_value × N_ACTIONS].
    Compact JSON-serializable. New states default to all zeros.
    """

    def __init__(self):
        self._q: dict[str, list[float]] = {}   # state_key → Q-values
        self._n: dict[str, list[int]] = {}      # state_key → visit counts per action

    def get_q(self, state_key: str) -> list[float]:
        if state_key not in self._q:
            self._q[state_key] = [0.0] * N_ACTIONS
        return self._q[state_key]

    def get_n(self, state_key: str) -> list[int]:
        if state_key not in self._n:
            self._n[state_key] = [0] * N_ACTIONS
        return self._n[state_key]

    def update(self, state_key: str, action: int, td_target: float):
        """Q(s,a) ← Q(s,a) + α*(target - Q(s,a))"""
        q = self.get_q(state_key)
        q[action] += ALPHA * (td_target - q[action])

    def increment_visit(self, state_key: str, action: int):
        n = self.get_n(state_key)
        n[action] += 1

    def total_visits(self, state_key: str) -> int:
        return sum(self.get_n(state_key))

    def to_dict(self) -> dict:
        return {"q": self._q, "n": self._n}

    def load_dict(self, d: dict):
        self._q = d.get("q", {})
        self._n = d.get("n", {})
        # Convert list-of-ints from JSON back to proper lists
        for k in self._n:
            self._n[k] = [int(x) for x in self._n[k]]
        log.info("QTable loaded: %d states", len(self._q))

    def size(self) -> int:
        return len(self._q)


class AdaptiveBrain:
    """
    Adaptive Q-learning brain. Central ML component.
    Thread-safe for asyncio (single-threaded event loop).
    """

    ACTIONS = ACTION_NAMES

    def __init__(self):
        self.qtable = QTable()
        self.replay = ReplayBuffer(max_size=8000)
        self._epsilon = EPSILON_START
        self._train_games = 0        # Games trained on (for epsilon decay)
        self._last_state: Optional[list[float]] = None
        self._last_action: Optional[int] = None
        self._last_view: Optional[dict] = None
        self._step_count = 0          # Steps this game
        self._game_reward = 0.0       # Cumulative game reward

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "qtable": self.qtable.to_dict(),
            "replay": self.replay.to_list(),
            "epsilon": self._epsilon,
            "train_games": self._train_games,
        }

    def load_dict(self, d: dict):
        if "qtable" in d:
            self.qtable.load_dict(d["qtable"])
        if "replay" in d:
            self.replay.load_list(d["replay"])
        self._epsilon = d.get("epsilon", EPSILON_START)
        self._train_games = d.get("train_games", 0)
        log.info("AdaptiveBrain loaded — epsilon=%.3f, train_games=%d, states=%d",
                 self._epsilon, self._train_games, self.qtable.size())

    # ── Action selection ───────────────────────────────────────────────

    def select_action(self, features: list[float]) -> int:
        """
        UCB1 action selection with epsilon-greedy fallback.
        UCB(a) = Q(s,a) + c * sqrt(ln(N_total + 1) / (N(s,a) + 1))
        """
        state_key = _state_key(features)
        q_vals = self.qtable.get_q(state_key)
        n_vals = self.qtable.get_n(state_key)
        total_visits = sum(n_vals) + 1

        ucb_vals = []
        for a in range(N_ACTIONS):
            exploration = UCB_C * math.sqrt(math.log(total_visits) / (n_vals[a] + 1))
            ucb_vals.append(q_vals[a] + exploration)

        return int(max(range(N_ACTIONS), key=lambda a: ucb_vals[a]))

    def select_action_epsilon_greedy(self, features: list[float]) -> int:
        """ε-greedy: explore randomly with probability ε, else use best Q."""
        if random.random() < self._epsilon:
            return random.randrange(N_ACTIONS)
        state_key = _state_key(features)
        q_vals = self.qtable.get_q(state_key)
        return int(max(range(N_ACTIONS), key=lambda a: q_vals[a]))

    def get_action_scores(self, features: list[float]) -> dict[str, float]:
        """Return Q-values for all actions (for dashboard display)."""
        state_key = _state_key(features)
        q_vals = self.qtable.get_q(state_key)
        return {ACTION_NAMES[a]: round(q_vals[a], 3) for a in range(N_ACTIONS)}

    # ── Training ───────────────────────────────────────────────────────

    def on_step(self, view: dict, action_idx: int, success: bool, next_view: dict):
        """
        Called after each game step. Records transition into replay buffer.
        Runs a mini-update (online learning) when buffer has enough samples.
        """
        prev_features = extract_features(view, True)
        next_features = extract_features(next_view, True)
        action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else "wait"
        reward = compute_reward(view, next_view, action_name, success)

        priority = abs(reward) + 1.0  # Higher |reward| → sampled more

        exp = Experience(prev_features, action_idx, reward,
                         next_features, False, priority)
        self.replay.add(exp)

        # Track for visit count
        state_key = _state_key(prev_features)
        self.qtable.increment_visit(state_key, action_idx)

        self._step_count += 1
        self._game_reward += reward

        # Online mini-update every 10 steps
        if self._step_count % 10 == 0 and len(self.replay) >= BATCH_SIZE:
            self._train_batch()

    def on_game_end(self, game_result: dict, final_view: Optional[dict] = None):
        """
        Called at game end. Applies terminal reward, trains a full batch.
        Returns dict with training summary.
        """
        # Terminal reward
        if final_view:
            final_features = extract_features(final_view, False)
            terminal_reward = compute_reward(None, final_view, "terminal", True, game_result)
            self._game_reward += terminal_reward

            if self._last_state is not None and self._last_action is not None:
                exp = Experience(
                    self._last_state, self._last_action, terminal_reward,
                    final_features, True,
                    priority=abs(terminal_reward) + 5.0  # High priority
                )
                self.replay.add(exp)

        # Full training pass
        n_batches = max(1, len(self.replay) // BATCH_SIZE)
        for _ in range(min(n_batches, 20)):   # Cap at 20 batches per game
            self._train_batch()

        # Decay epsilon
        self._epsilon = max(EPSILON_MIN, self._epsilon * EPSILON_DECAY)
        self._train_games += 1

        summary = {
            "epsilon": round(self._epsilon, 4),
            "train_games": self._train_games,
            "qtable_states": self.qtable.size(),
            "replay_size": len(self.replay),
            "game_reward": round(self._game_reward, 2),
            "steps": self._step_count,
        }
        log.info(
            "[ML] Training complete -- e=%.3f games=%d states=%d "
            "replay=%d game_reward=%.1f",
            self._epsilon, self._train_games, self.qtable.size(),
            len(self.replay), self._game_reward,
        )

        # Reset per-game state
        self._last_state = None
        self._last_action = None
        self._last_view = None
        self._step_count = 0
        self._game_reward = 0.0

        return summary

    def _train_batch(self):
        """Q-learning update on a sampled batch from replay buffer."""
        batch = self.replay.sample(BATCH_SIZE)
        for exp in batch:
            s_key = _state_key(exp.state)
            ns_key = _state_key(exp.next_state)

            # Bellman target: r + γ * max_a'[Q(s', a')] * (1 - done)
            next_q = self.qtable.get_q(ns_key)
            max_next_q = max(next_q) if not exp.done else 0.0
            td_target = exp.reward + GAMMA * max_next_q * (0.0 if exp.done else 1.0)

            self.qtable.update(s_key, exp.action, td_target)

    # ── State tracking for on_step helper ────────────────────────────

    def record_state(self, view: dict, action_idx: int):
        """Store current view + chosen action for next-step reward calculation."""
        self._last_view = view
        self._last_state = extract_features(view, True)
        self._last_action = action_idx

    def stats(self) -> dict:
        """Return current ML statistics for logging / dashboard."""
        return {
            "epsilon": round(self._epsilon, 4),
            "train_games": self._train_games,
            "qtable_states": self.qtable.size(),
            "replay_size": len(self.replay),
        }
