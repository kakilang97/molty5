"""Unit tests for the adaptive learning policy.

These tests are pure-Python (no network, no websocket, no game server).
They guard the algorithm-level invariants the rule-based brain relies
on: state featurization is deterministic, the Q update has the right
direction and magnitude, the bandit prefers higher-reward arms, and the
JSON round-trip is lossless.
"""
from __future__ import annotations

import math
import random

import pytest

from bot.strategy.adaptive import (
    AdaptiveLearner,
    BanditArm,
    ContextualBandit,
    COMBAT_HP_THRESHOLDS,
    HEAL_HP_THRESHOLDS,
    MACROS,
    featurize_state,
    state_key,
)


# ── Featurization ────────────────────────────────────────────────────


def _view(
    hp=80, ep=8, max_ep=10,
    alive=50, region_id="r1", in_dz=False,
    pending=(), enemies=0, guardians=0, monsters=0,
    inventory=None, weapon=None,
):
    visible_agents = []
    for i in range(enemies):
        visible_agents.append({
            "id": f"e{i}", "isAlive": True, "regionId": region_id,
            "isGuardian": False, "hp": 50,
        })
    for i in range(guardians):
        visible_agents.append({
            "id": f"g{i}", "isAlive": True, "regionId": region_id,
            "isGuardian": True, "hp": 80,
        })
    visible_monsters = [{"id": f"m{i}", "hp": 30} for i in range(monsters)]
    return {
        "self": {
            "id": "me", "hp": hp, "ep": ep, "maxEp": max_ep,
            "isAlive": True,
            "inventory": inventory or [],
            "equippedWeapon": weapon,
        },
        "currentRegion": {"id": region_id, "isDeathZone": in_dz},
        "visibleAgents": visible_agents,
        "visibleMonsters": visible_monsters,
        "pendingDeathzones": [{"id": p, "name": p} for p in pending],
        "aliveCount": alive,
    }


def test_featurize_is_deterministic():
    v = _view()
    assert featurize_state(v) == featurize_state(v)


def test_featurize_buckets_low_hp_separately_from_high_hp():
    low = featurize_state(_view(hp=10))
    high = featurize_state(_view(hp=95))
    assert low[0] != high[0]


def test_featurize_threat_grows_with_enemies():
    s0 = featurize_state(_view(enemies=0, guardians=0))
    s2 = featurize_state(_view(enemies=2, guardians=0))
    s5 = featurize_state(_view(enemies=4, guardians=2))
    assert s0[3] <= s2[3] <= s5[3]


def test_featurize_marks_death_zone():
    safe = featurize_state(_view(in_dz=False))
    dz = featurize_state(_view(in_dz=True))
    pending = featurize_state(_view(in_dz=False, pending=("r1",)))
    # danger flag is the last tuple slot
    assert safe[-1] == 0
    assert pending[-1] == 1
    assert dz[-1] == 2


def test_featurize_weapon_tier():
    fist = featurize_state(_view(weapon=None))
    katana = featurize_state(_view(weapon={"typeId": "katana"}))
    assert katana[5] > fist[5]


def test_state_key_is_stringy_and_stable():
    s = featurize_state(_view())
    k = state_key(s)
    assert isinstance(k, str)
    assert k == state_key(s)


# ── Q-learning update ────────────────────────────────────────────────


def test_q_update_moves_toward_target():
    learner = AdaptiveLearner(alpha=0.5, gamma=0.9)
    learner.epsilon_init = 0.0
    learner.epsilon_floor = 0.0  # pure exploit for determinism
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    s2 = (1, 1, 1, 1, 1, 1, 1, 0)
    # Seed Q(s2, *) so target reward + gamma*max_a Q(s2, a) is non-zero.
    learner._q_row(state_key(s2))["engage_player"] = 10.0
    # Q(s, a) before update is 0 → target = r + γ * 10 = 1 + 9 = 10
    # New Q(s, a) = 0 + α * (10 - 0) = 5
    learner.observe_step(s, "engage_player", reward=1.0, next_state=s2)
    assert learner._q_row(state_key(s))["engage_player"] == pytest.approx(5.0)


def test_q_update_terminal_uses_only_reward():
    learner = AdaptiveLearner(alpha=1.0, gamma=0.9)
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    learner.observe_step(s, "engage_player", 7.5, s, done=True)
    assert learner._q_row(state_key(s))["engage_player"] == pytest.approx(7.5)


def test_choose_macro_respects_allowed_set():
    learner = AdaptiveLearner()
    learner.epsilon_init = 0.0
    learner.epsilon_floor = 0.0
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    # Force a single allowed macro — must always be returned.
    chosen = [learner.choose_macro(s, allowed=["rest"]) for _ in range(20)]
    assert all(m == "rest" for m in chosen)


def test_choose_macro_exploits_best_known_value():
    learner = AdaptiveLearner()
    learner.epsilon_init = 0.0
    learner.epsilon_floor = 0.0
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    row = learner._q_row(state_key(s))
    row["engage_player"] = 5.0
    row["flee"] = -1.0
    assert learner.choose_macro(s, allowed=list(MACROS)) == "engage_player"


def test_choose_macro_explores_when_epsilon_high():
    random.seed(0)
    learner = AdaptiveLearner()
    learner.epsilon_init = 1.0
    learner.epsilon_floor = 1.0
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    row = learner._q_row(state_key(s))
    row["engage_player"] = 100.0
    # With ε=1 we should not always pick the greedy choice.
    picks = {learner.choose_macro(s, list(MACROS)) for _ in range(50)}
    assert len(picks) > 1


# ── Reward shaping ───────────────────────────────────────────────────


def test_step_reward_kills_and_hp_loss():
    prev = {
        "self": {"hp": 100, "kills": 0, "isAlive": True, "smoltz": 0},
        "aliveCount": 50,
    }
    cur = {
        "self": {"hp": 80, "kills": 1, "isAlive": True, "smoltz": 100},
        "aliveCount": 49,
    }
    r = AdaptiveLearner.step_reward(prev, cur)
    # +1 kill, -20 HP, +100 smoltz, -1 alive_count, +survival
    assert r > 0  # Positive overall — kill+smoltz dominates the HP loss
    assert r < 20  # but bounded


def test_step_reward_death_is_strongly_negative():
    prev = {"self": {"hp": 30, "isAlive": True}, "aliveCount": 50}
    cur = {"self": {"hp": 0, "isAlive": False}, "aliveCount": 50}
    assert AdaptiveLearner.step_reward(prev, cur) < -5


def test_terminal_reward_winner_dominates():
    win = AdaptiveLearner.terminal_reward(
        is_winner=True, final_rank=1, kills=3, smoltz_earned=500
    )
    lose = AdaptiveLearner.terminal_reward(
        is_winner=False, final_rank=80, kills=0, smoltz_earned=0
    )
    assert win > lose
    assert win > 50


# ── Bandit ───────────────────────────────────────────────────────────


def test_bandit_first_explores_each_arm_once():
    b = ContextualBandit("t", arms=COMBAT_HP_THRESHOLDS)
    seen = set()
    for _ in range(len(COMBAT_HP_THRESHOLDS)):
        a = b.select()
        b.update(a, reward=0.0)
        seen.add(a)
    assert seen == set(COMBAT_HP_THRESHOLDS)


def test_bandit_prefers_higher_reward_arm_long_run():
    random.seed(42)
    b = ContextualBandit("t", arms=COMBAT_HP_THRESHOLDS)
    # Reward function: arm 40 is best.
    rewards = {25: 1.0, 40: 5.0, 55: 2.0}
    for _ in range(500):
        a = b.select()
        b.update(a, reward=rewards[a])
    means = {a: b.arm_stats[a].mean for a in COMBAT_HP_THRESHOLDS}
    best = max(means, key=means.get)
    assert best == 40
    # And the best arm must have been pulled significantly more often.
    assert b.arm_stats[40].n > b.arm_stats[25].n


# ── Persistence ──────────────────────────────────────────────────────


def test_round_trip_preserves_q_values_and_bandit_stats():
    learner = AdaptiveLearner(alpha=0.3, gamma=0.95)
    s = (0, 1, 2, 1, 0, 2, 1, 0)
    learner.observe_step(s, "engage_player", 1.0, s, done=True)
    learner.combat_hp_bandit.update(40, 12.5)
    learner.heal_hp_bandit.update(50, 7.0)
    learner.games_played = 4
    blob = learner.to_dict()
    restored = AdaptiveLearner.from_dict(blob)
    assert restored.alpha == pytest.approx(0.3)
    assert restored.gamma == pytest.approx(0.95)
    assert restored.games_played == 4
    assert restored._q_row(state_key(s))["engage_player"] == pytest.approx(
        learner._q_row(state_key(s))["engage_player"]
    )
    assert restored.combat_hp_bandit.arm_stats[40].n == 1
    assert restored.combat_hp_bandit.arm_stats[40].reward_sum == pytest.approx(12.5)
    assert restored.heal_hp_bandit.arm_stats[50].n == 1


def test_round_trip_through_json_module():
    import json
    learner = AdaptiveLearner()
    s = (0, 0, 0, 0, 0, 0, 0, 0)
    learner.observe_step(s, "rest", 0.5, s, done=True)
    serialized = json.dumps(learner.to_dict())
    restored = AdaptiveLearner.from_dict(json.loads(serialized))
    assert restored._q_row(state_key(s))["rest"] == pytest.approx(
        learner._q_row(state_key(s))["rest"]
    )


def test_epsilon_decays_over_games():
    learner = AdaptiveLearner()
    learner.epsilon_init = 0.30
    learner.epsilon_floor = 0.05
    e0 = learner.epsilon
    learner.games_played = 100
    e_late = learner.epsilon
    assert e0 > e_late
    assert math.isclose(e_late, 0.05, abs_tol=1e-6)
