"""
Adaptive learning policy — real online RL on top of the rule-based brain.

Design (kept lightweight so it runs on Railway / small Docker containers):

1. Tabular Q-learning over a discretized state, with macro actions for
   discretionary decisions (combat, healing, exploration). Hard-safety
   behaviours (death-zone escape, free pickups, equip, use Map) stay in
   `brain.py` — those are not allowed to drift via learning.

2. Per-game contextual bandits adapt the two most impactful continuous
   thresholds (`combat_hp_threshold`, `heal_hp_threshold`).  At the start
   of each game we sample a value with UCB1; after the game we feed back
   the score (sMoltz earned + ranking points + survived turns).

3. The whole policy state lives in the existing
   `molty-royale-context.json` memory file under a new `policy` section,
   so it persists across container restarts and Railway redeploys.

4. Pure Python, no numpy / torch — keeps the deploy footprint small.

The integration points are:

    adaptive = AdaptiveLearner.from_dict(memory["policy"])
    state = adaptive.featurize(view)
    macro = adaptive.choose_macro(state)
    # ... brain realises the macro into a concrete action ...
    adaptive.observe_step(state, macro, reward, next_state)
    adaptive.observe_game_end(final_reward, sMoltz, rank, kills, won)
    memory["policy"] = adaptive.to_dict()
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Iterable

from bot.utils.logger import get_logger

log = get_logger(__name__)

# ── Macro actions ─────────────────────────────────────────────────────
# These are coarse, high-level intents.  brain.py is responsible for
# turning the chosen macro into a concrete game action; if it cannot
# realise the macro (e.g. ENGAGE_PLAYER but no enemy in range) it falls
# back to existing rule-based behaviour.
MACROS: tuple[str, ...] = (
    "engage_player",
    "engage_guardian",
    "farm_monster",
    "flee",
    "heal",
    "rest",
    "move_explore",
    "interact",
)

# ── Continuous-threshold bandit arms ──────────────────────────────────
COMBAT_HP_THRESHOLDS = (25, 40, 55)
HEAL_HP_THRESHOLDS = (30, 50, 70)

# ── Hyperparameters (overridable via env in config.py) ────────────────
DEFAULT_ALPHA = 0.20      # learning rate
DEFAULT_GAMMA = 0.90      # discount
DEFAULT_EPSILON_INIT = 0.30
DEFAULT_EPSILON_FLOOR = 0.05
EPSILON_DECAY_GAMES = 100  # games to fully decay from init -> floor

# Reward shaping coefficients (tunable, but documented as defaults).
REWARD_PER_KILL = 5.0
REWARD_PER_SMOLTZ = 0.05
REWARD_PER_HP_GAIN = 0.10
REWARD_PER_HP_LOSS = -0.10
REWARD_PER_OPP_DEATH = 0.20  # alive_count drop attributable to others
REWARD_SURVIVAL_TICK = 0.05
REWARD_DEATH = -10.0
REWARD_WIN = 50.0


# ── State featurizer ──────────────────────────────────────────────────

def _bucket(value: float, edges: Iterable[float]) -> int:
    """Return the index of the first edge >= value, else len(edges)."""
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(list(edges))


def featurize_state(view: dict) -> tuple:
    """Discretize a raw view into a hashable state tuple.

    Buckets are chosen to keep the table small (<10k cells) while still
    capturing the qualitatively distinct situations the bot encounters.
    """
    self_data = view.get("self", {}) or {}
    region = view.get("currentRegion", {}) or {}
    visible_agents = view.get("visibleAgents", []) or []
    visible_monsters = view.get("visibleMonsters", []) or []
    pending_dz = view.get("pendingDeathzones", []) or []
    inventory = self_data.get("inventory", []) or []
    equipped = self_data.get("equippedWeapon")

    hp = int(self_data.get("hp", 100) or 100)
    ep = int(self_data.get("ep", 0) or 0)
    max_ep = max(1, int(self_data.get("maxEp", 10) or 10))
    alive = int(view.get("aliveCount", 100) or 100)

    hp_bucket = _bucket(hp, (25, 50, 75, 100))             # 0..3
    ep_bucket = _bucket(ep / max_ep, (0.2, 0.5, 0.8, 1.0))  # 0..3
    alive_bucket = _bucket(alive, (5, 15, 40, 100))         # 0..3 (lower=late game)

    region_id = region.get("id", "") if isinstance(region, dict) else ""
    in_dz = bool(region.get("isDeathZone")) if isinstance(region, dict) else False
    pending_ids = set()
    for dz in pending_dz:
        if isinstance(dz, dict):
            pending_ids.add(dz.get("id", ""))
        elif isinstance(dz, str):
            pending_ids.add(dz)
    in_pending = region_id in pending_ids
    danger_flag = 2 if in_dz else (1 if in_pending else 0)

    # threat: hostile agents in same region
    enemies_here = sum(
        1 for a in visible_agents
        if isinstance(a, dict) and a.get("isAlive", True)
        and a.get("regionId") == region_id
        and a.get("id") != self_data.get("id")
    )
    guardians_here = sum(
        1 for a in visible_agents
        if isinstance(a, dict) and a.get("isAlive", True)
        and a.get("isGuardian", False)
        and a.get("regionId") == region_id
    )
    threat_bucket = _bucket(enemies_here + guardians_here, (0, 1, 3))  # 0..3

    monsters_here = sum(
        1 for m in visible_monsters
        if isinstance(m, dict) and (m.get("hp", 0) or 0) > 0
    )
    has_monster = 1 if monsters_here > 0 else 0

    # weapon tier (0=fist, 1=dagger/bow, 2=sword/pistol, 3=katana/sniper)
    weapon_tier = 0
    if isinstance(equipped, dict):
        type_id = (equipped.get("typeId") or "").lower()
        weapon_tier = {
            "fist": 0,
            "dagger": 1, "bow": 1,
            "sword": 2, "pistol": 2,
            "katana": 3, "sniper": 3,
        }.get(type_id, 0)

    # heal stockpile (0,1,2+)
    heal_count = 0
    for it in inventory:
        if not isinstance(it, dict):
            continue
        tid = (it.get("typeId") or "").lower()
        if tid in {"medkit", "bandage", "emergency_food"}:
            heal_count += 1
    heal_bucket = min(heal_count, 2)

    return (
        hp_bucket, ep_bucket, alive_bucket,
        threat_bucket, has_monster,
        weapon_tier, heal_bucket, danger_flag,
    )


def state_key(state: tuple) -> str:
    """Stable string form for JSON serialization."""
    return ",".join(str(x) for x in state)


# ── Bandit arm tracker ────────────────────────────────────────────────

@dataclass
class BanditArm:
    n: int = 0
    reward_sum: float = 0.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.n if self.n > 0 else 0.0


@dataclass
class ContextualBandit:
    """UCB1 over a small fixed set of arms."""
    name: str
    arms: tuple[int, ...]
    arm_stats: dict[int, BanditArm] = field(default_factory=dict)
    total_pulls: int = 0

    def __post_init__(self):
        for a in self.arms:
            self.arm_stats.setdefault(a, BanditArm())

    def select(self) -> int:
        # Try each arm at least once
        for a in self.arms:
            if self.arm_stats[a].n == 0:
                return a
        # UCB1
        log_total = math.log(max(1, self.total_pulls))
        best, best_score = self.arms[0], -float("inf")
        for a in self.arms:
            stats = self.arm_stats[a]
            bonus = math.sqrt(2.0 * log_total / max(1, stats.n))
            score = stats.mean + bonus
            if score > best_score:
                best, best_score = a, score
        return best

    def update(self, arm: int, reward: float):
        stats = self.arm_stats.setdefault(arm, BanditArm())
        stats.n += 1
        stats.reward_sum += reward
        self.total_pulls += 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arms": list(self.arms),
            "total_pulls": self.total_pulls,
            "stats": {
                str(a): [self.arm_stats[a].n, self.arm_stats[a].reward_sum]
                for a in self.arms
            },
        }

    @classmethod
    def from_dict(cls, data: dict, default_arms: tuple[int, ...]) -> "ContextualBandit":
        arms = tuple(data.get("arms", default_arms)) if data else default_arms
        b = cls(name=(data or {}).get("name", "bandit"), arms=arms)
        b.total_pulls = int((data or {}).get("total_pulls", 0))
        for a in arms:
            row = ((data or {}).get("stats", {}) or {}).get(str(a), [0, 0.0])
            n, rs = (row + [0, 0.0])[:2]
            b.arm_stats[a] = BanditArm(n=int(n), reward_sum=float(rs))
        return b


# ── Q-learning agent ──────────────────────────────────────────────────

@dataclass
class AdaptiveLearner:
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    epsilon_init: float = DEFAULT_EPSILON_INIT
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR
    games_played: int = 0
    training_step: int = 0
    q_table: dict[str, dict[str, float]] = field(default_factory=dict)
    visit_count: dict[str, int] = field(default_factory=dict)
    combat_hp_bandit: ContextualBandit = field(
        default_factory=lambda: ContextualBandit("combat_hp", COMBAT_HP_THRESHOLDS)
    )
    heal_hp_bandit: ContextualBandit = field(
        default_factory=lambda: ContextualBandit("heal_hp", HEAL_HP_THRESHOLDS)
    )
    last_combat_arm: int = 40
    last_heal_arm: int = 50
    last_save_step: int = 0
    enabled: bool = True

    # ── policy ────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        if EPSILON_DECAY_GAMES <= 0:
            return self.epsilon_floor
        progress = min(1.0, self.games_played / EPSILON_DECAY_GAMES)
        return self.epsilon_init + (self.epsilon_floor - self.epsilon_init) * progress

    def _q_row(self, key: str) -> dict[str, float]:
        row = self.q_table.get(key)
        if row is None:
            row = {m: 0.0 for m in MACROS}
            self.q_table[key] = row
        else:
            # backfill new macros if MACROS expanded since last save
            for m in MACROS:
                row.setdefault(m, 0.0)
        return row

    def choose_macro(self, state: tuple, allowed: Iterable[str] | None = None) -> str:
        """ε-greedy over Q.  `allowed` lets brain.py mask out macros that
        cannot be realised in the current view (e.g. no enemies → no
        engage)."""
        key = state_key(state)
        row = self._q_row(key)
        candidates = list(allowed) if allowed else list(MACROS)
        if not candidates:
            return "rest"
        if random.random() < self.epsilon:
            return random.choice(candidates)
        # exploit
        best, best_q = candidates[0], -float("inf")
        for m in candidates:
            q = row.get(m, 0.0)
            if q > best_q:
                best, best_q = m, q
        return best

    def observe_step(
        self,
        prev_state: tuple,
        macro: str,
        reward: float,
        next_state: tuple,
        done: bool = False,
    ) -> None:
        """Standard Q-learning update."""
        if not self.enabled:
            return
        prev_key = state_key(prev_state)
        next_key = state_key(next_state)
        row = self._q_row(prev_key)
        q_sa = row.get(macro, 0.0)
        if done:
            target = reward
        else:
            next_row = self._q_row(next_key)
            target = reward + self.gamma * max(next_row.values())
        row[macro] = q_sa + self.alpha * (target - q_sa)
        self.visit_count[prev_key] = self.visit_count.get(prev_key, 0) + 1
        self.training_step += 1

    # ── per-game bandit helpers ───────────────────────────────────────

    def start_game(self) -> None:
        """Sample thresholds for the upcoming game."""
        self.last_combat_arm = self.combat_hp_bandit.select()
        self.last_heal_arm = self.heal_hp_bandit.select()
        log.info(
            "🧠 Adaptive policy: ε=%.2f, combat_hp=%d, heal_hp=%d, games=%d, q_states=%d",
            self.epsilon, self.last_combat_arm, self.last_heal_arm,
            self.games_played, len(self.q_table),
        )

    def observe_game_end(
        self,
        bandit_reward: float,
    ) -> None:
        """Update bandit arms with the game's overall score."""
        if not self.enabled:
            return
        self.combat_hp_bandit.update(self.last_combat_arm, bandit_reward)
        self.heal_hp_bandit.update(self.last_heal_arm, bandit_reward)
        self.games_played += 1

    # ── reward shaping helpers ────────────────────────────────────────

    @staticmethod
    def step_reward(prev_view: dict | None, view: dict) -> float:
        """Per-turn reward from observation deltas."""
        if not view:
            return 0.0
        self_now = view.get("self", {}) or {}
        if prev_view is None:
            # First view of a game — small survival bonus only.
            return REWARD_SURVIVAL_TICK if self_now.get("isAlive", True) else REWARD_DEATH

        self_prev = prev_view.get("self", {}) or {}
        r = 0.0

        hp_now = float(self_now.get("hp", 0) or 0)
        hp_prev = float(self_prev.get("hp", 0) or 0)
        dhp = hp_now - hp_prev
        if dhp > 0:
            r += REWARD_PER_HP_GAIN * dhp
        elif dhp < 0:
            r += REWARD_PER_HP_LOSS * abs(dhp)

        kills_now = int(self_now.get("kills", 0) or 0)
        kills_prev = int(self_prev.get("kills", 0) or 0)
        r += REWARD_PER_KILL * max(0, kills_now - kills_prev)

        smoltz_now = float(
            self_now.get("smoltz",
                         self_now.get("balance",
                                      view.get("balance", 0))) or 0
        )
        smoltz_prev = float(
            self_prev.get("smoltz",
                          self_prev.get("balance",
                                        prev_view.get("balance", 0))) or 0
        )
        r += REWARD_PER_SMOLTZ * max(0.0, smoltz_now - smoltz_prev)

        alive_now = int(view.get("aliveCount", 100) or 100)
        alive_prev = int(prev_view.get("aliveCount", alive_now) or alive_now)
        # If alive count dropped and we are still alive, others died → good.
        if self_now.get("isAlive", True) and alive_now < alive_prev:
            r += REWARD_PER_OPP_DEATH * (alive_prev - alive_now)

        if not self_now.get("isAlive", True) and self_prev.get("isAlive", True):
            r += REWARD_DEATH

        if self_now.get("isAlive", True):
            r += REWARD_SURVIVAL_TICK
        return r

    @staticmethod
    def terminal_reward(
        is_winner: bool,
        final_rank: int,
        kills: int,
        smoltz_earned: float,
    ) -> float:
        r = 0.0
        if is_winner:
            r += REWARD_WIN
        if final_rank > 0:
            r += max(0.0, 30.0 - final_rank)  # top-30 bonus, decays
        r += REWARD_PER_KILL * max(0, kills)
        r += REWARD_PER_SMOLTZ * max(0.0, smoltz_earned)
        if final_rank >= 50 and not is_winner:
            r += REWARD_DEATH
        return r

    # ── persistence ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_init": self.epsilon_init,
            "epsilon_floor": self.epsilon_floor,
            "games_played": self.games_played,
            "training_step": self.training_step,
            "q_table": self.q_table,
            "visit_count": self.visit_count,
            "bandit": {
                "combat_hp_threshold": self.combat_hp_bandit.to_dict(),
                "heal_hp_threshold": self.heal_hp_bandit.to_dict(),
            },
            "last_arms": {
                "combat_hp": self.last_combat_arm,
                "heal_hp": self.last_heal_arm,
            },
            "updated_at": int(time.time()),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "AdaptiveLearner":
        data = data or {}
        learner = cls(
            alpha=float(data.get("alpha", DEFAULT_ALPHA)),
            gamma=float(data.get("gamma", DEFAULT_GAMMA)),
            epsilon_init=float(data.get("epsilon_init", DEFAULT_EPSILON_INIT)),
            epsilon_floor=float(data.get("epsilon_floor", DEFAULT_EPSILON_FLOOR)),
            games_played=int(data.get("games_played", 0)),
            training_step=int(data.get("training_step", 0)),
        )
        # q_table may serialize numbers as JSON numbers — coerce to float.
        raw_q = data.get("q_table", {}) or {}
        for k, row in raw_q.items():
            learner.q_table[k] = {m: float(row.get(m, 0.0)) for m in MACROS}
            for extra_macro, val in row.items():
                if extra_macro not in learner.q_table[k]:
                    # forward-compat: keep any unknown macro values
                    learner.q_table[k][extra_macro] = float(val)
        learner.visit_count = {
            k: int(v) for k, v in (data.get("visit_count", {}) or {}).items()
        }
        bandits = data.get("bandit", {}) or {}
        learner.combat_hp_bandit = ContextualBandit.from_dict(
            bandits.get("combat_hp_threshold"), COMBAT_HP_THRESHOLDS
        )
        learner.heal_hp_bandit = ContextualBandit.from_dict(
            bandits.get("heal_hp_threshold"), HEAL_HP_THRESHOLDS
        )
        last = data.get("last_arms", {}) or {}
        learner.last_combat_arm = int(last.get("combat_hp", 40))
        learner.last_heal_arm = int(last.get("heal_hp", 50))
        return learner

    # ── introspection ────────────────────────────────────────────────

    def policy_snapshot(self, state: tuple, top_n: int = 3) -> list[tuple[str, float]]:
        """Return top-N (macro, q) pairs for the given state — for dashboards."""
        row = self._q_row(state_key(state))
        return sorted(row.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
