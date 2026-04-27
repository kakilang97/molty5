"""
Experience Replay Buffer — stores (state, action, reward, next_state, done) tuples.

Used by the adaptive Q-learning brain to train on past game transitions.
Implements prioritized sampling: more recent AND higher-reward experiences
are sampled more often, while maintaining a ring buffer (no unbounded growth).

Storage: serializable to JSON for persistence across sessions.
"""
import json
import random
import math
from pathlib import Path
from bot.utils.logger import get_logger

log = get_logger(__name__)

# Action index mapping — must stay in sync with AdaptiveBrain.ACTIONS
ACTION_NAMES = [
    "escape_dz",         # 0 — death zone escape (highest priority)
    "pre_escape",        # 1 — escape before DZ activates
    "flee_guardian",     # 2 — run from guardian when low HP
    "pickup",            # 3 — pick up item
    "equip",             # 4 — equip better weapon
    "use_utility",       # 5 — use map/megaphone
    "heal_critical",     # 6 — critical heal (HP < 30)
    "heal_moderate",     # 7 — moderate heal (HP < 70)
    "ep_recovery",       # 8 — use energy drink
    "attack_guardian",   # 9 — kill guardian for sMoltz
    "attack_enemy",      # 10 — fight another player
    "attack_monster",    # 11 — farm monster
    "use_facility",      # 12 — interact with facility
    "move",              # 13 — strategic movement
    "rest",              # 14 — rest to recover EP
    "wait",              # 15 — skip (no action)
]

N_ACTIONS = len(ACTION_NAMES)


class Experience:
    """A single SARS+ tuple."""
    __slots__ = ("state", "action", "reward", "next_state", "done", "priority")

    def __init__(self, state: list, action: int, reward: float,
                 next_state: list, done: bool, priority: float = 1.0):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.priority = priority

    def to_dict(self) -> dict:
        return {
            "s": self.state, "a": self.action, "r": self.reward,
            "ns": self.next_state, "d": self.done, "p": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        return cls(d["s"], d["a"], d["r"], d["ns"], d["d"], d.get("p", 1.0))


class ReplayBuffer:
    """
    Prioritized ring buffer for experience replay.
    Max capacity = max_size. When full, evicts oldest (lowest-priority) entry.
    Priority = |reward| + recency_bonus, so high-magnitude transitions (wins/deaths)
    are always kept and sampled more often.
    """

    def __init__(self, max_size: int = 5000):
        self._buf: list[Experience] = []
        self._max_size = max_size

    def add(self, exp: Experience):
        """Add experience. Evict lowest-priority when full."""
        if len(self._buf) >= self._max_size:
            # Remove lowest priority item
            idx = min(range(len(self._buf)), key=lambda i: self._buf[i].priority)
            self._buf.pop(idx)
        self._buf.append(exp)

    def sample(self, n: int) -> list[Experience]:
        """Sample n experiences weighted by priority."""
        if len(self._buf) == 0:
            return []
        n = min(n, len(self._buf))
        weights = [e.priority for e in self._buf]
        return random.choices(self._buf, weights=weights, k=n)

    def __len__(self) -> int:
        return len(self._buf)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self._buf]

    def load_list(self, data: list[dict]):
        self._buf = [Experience.from_dict(d) for d in data]
        log.info("ReplayBuffer loaded %d experiences", len(self._buf))


def compute_reward(prev_view: dict, curr_view: dict, action_type: str,
                   action_success: bool, game_result: dict | None = None) -> float:
    """
    Reward shaping for Molty Royale:
    - Terminal rewards dominate: +100 win, -30 death
    - Per-step rewards guide toward good decisions
    - Reward is clipped to [-10, +10] for intermediate steps to keep Q-values stable
    """
    if game_result:
        is_winner = game_result.get("isWinner", False) or game_result.get("winner") is not None
        kills = game_result.get("kills", 0) or game_result.get("agentKills", 0)
        smoltz = game_result.get("smoltzEarned", 0) or game_result.get("sMoltzEarned", 0)
        rank = game_result.get("finalRank", 99) or game_result.get("rank", 99)

        if is_winner:
            return 100.0 + kills * 5.0 + smoltz / 100.0
        elif rank <= 3:
            return 30.0 + kills * 3.0
        elif rank <= 10:
            return 10.0 + kills * 2.0
        else:
            return -20.0 + kills * 2.0 + smoltz / 200.0

    if not action_success:
        return -0.5   # Failed action is wasteful

    # Pull state from views
    prev_self = prev_view.get("self", {}) if prev_view else {}
    curr_self = curr_view.get("self", {}) if curr_view else {}

    prev_hp = prev_self.get("hp", 100)
    curr_hp = curr_self.get("hp", prev_hp)
    prev_ep = prev_self.get("ep", 10)
    curr_ep = curr_self.get("ep", prev_ep)
    prev_kills = prev_self.get("kills", 0)
    curr_kills = curr_self.get("kills", prev_kills)
    prev_smoltz = prev_self.get("sMoltz", 0) or prev_self.get("balance", 0)
    curr_smoltz = curr_self.get("sMoltz", 0) or curr_self.get("balance", prev_smoltz)
    in_dz = curr_view.get("currentRegion", {}).get("isDeathZone", False) if curr_view else False
    prev_dz = prev_view.get("currentRegion", {}).get("isDeathZone", False) if prev_view else False
    is_dead = not curr_self.get("isAlive", True)
    alive_count = curr_view.get("aliveCount", 100) if curr_view else 100

    reward = 0.0

    # Death penalty
    if is_dead:
        reward -= 30.0
        return reward

    # DZ penalty / escape bonus
    if in_dz and not prev_dz:
        reward -= 3.0       # Entered a DZ — bad
    elif not in_dz and prev_dz:
        reward += 5.0       # Escaped a DZ — good

    # Kill reward
    kill_delta = curr_kills - prev_kills
    if kill_delta > 0:
        reward += kill_delta * 8.0   # Each kill is valuable

    # sMoltz pickup
    smoltz_delta = curr_smoltz - prev_smoltz
    if smoltz_delta > 0:
        reward += smoltz_delta * 0.05

    # HP change
    hp_delta = curr_hp - prev_hp
    if hp_delta > 0:
        reward += hp_delta * 0.1    # Healing is good
    elif hp_delta < -10:
        reward += hp_delta * 0.05   # Taking heavy damage is bad

    # Penalize inaction when threatened
    if action_type == "wait" and in_dz:
        reward -= 5.0

    # Survival bonus — just staying alive as alive_count drops is valuable
    if alive_count < 20:
        reward += 0.5   # Late-game survival bonus

    # Clip intermediate rewards
    return max(-10.0, min(10.0, reward))
