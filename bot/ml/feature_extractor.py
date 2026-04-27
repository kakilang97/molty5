"""
Feature Extractor — converts raw agent_view into a normalized numeric feature vector.

Features are designed to capture all strategically-relevant game state:
- Survival metrics: HP%, EP%, alive_count_normalized
- Threat assessment: nearest_enemy_hp%, guardian_nearby, enemy_count
- Positioning: is_in_dz, pending_dz_nearby, safe_exits_count
- Combat readiness: weapon_bonus_normalized, can_act
- Resource state: heal_count, has_binos, inventory_space
- Late-game indicator: alive ratio (1.0 = just started, 0.0 = final duel)

Total: 20 features, all normalized to [0, 1].
"""
from bot.utils.logger import get_logger

log = get_logger(__name__)

# Weapon bonus range [0..35] — used for normalization
MAX_WEAPON_BONUS = 35
MAX_HP = 100
MAX_EP = 10
MAX_ALIVE = 100


def extract_features(view: dict, can_act: bool) -> list[float]:
    """
    Convert a raw game view dict into a 20-dimensional feature vector.
    All values are normalized to [0.0, 1.0].
    Returns zeros if view is invalid — safe to call on partial states.
    """
    try:
        return _extract(view, can_act)
    except Exception as e:
        log.warning("Feature extraction failed: %s — returning zeros", e)
        return [0.0] * 20


def _extract(view: dict, can_act: bool) -> list[float]:
    self_data = view.get("self", {})
    region = view.get("currentRegion", {}) if isinstance(view.get("currentRegion"), dict) else {}
    visible_agents = view.get("visibleAgents", [])
    visible_monsters = view.get("visibleMonsters", [])
    inventory = self_data.get("inventory", [])
    pending_dz = view.get("pendingDeathzones", [])
    connected = view.get("connectedRegions", [])
    alive_count = view.get("aliveCount", MAX_ALIVE)

    hp = self_data.get("hp", MAX_HP)
    ep = self_data.get("ep", MAX_EP)
    max_hp = self_data.get("maxHp", MAX_HP)
    max_ep = self_data.get("maxEp", MAX_EP)
    atk = self_data.get("atk", 10)
    defense = self_data.get("def", 5)
    equipped = self_data.get("equippedWeapon")

    region_id = region.get("id", "")
    is_dz = float(region.get("isDeathZone", False))
    terrain = region.get("terrain", "").lower()
    weather = region.get("weather", "").lower()

    # ── Feature 0: HP ratio ──────────────────────────────────────────
    f_hp = _clamp(hp / max(max_hp, 1))

    # ── Feature 1: EP ratio ──────────────────────────────────────────
    f_ep = _clamp(ep / max(max_ep, 1))

    # ── Feature 2: alive ratio (late-game indicator) ─────────────────
    f_alive_ratio = _clamp(alive_count / MAX_ALIVE)

    # ── Feature 3: in death zone ─────────────────────────────────────
    f_is_dz = is_dz

    # ── Feature 4: pending DZ nearby ─────────────────────────────────
    pending_ids = set()
    for dz in pending_dz:
        if isinstance(dz, dict):
            pending_ids.add(dz.get("id", ""))
        elif isinstance(dz, str):
            pending_ids.add(dz)
    f_pending_dz = _clamp(len(pending_ids) / 5.0)

    # ── Feature 5: number of safe exits ─────────────────────────────
    safe_exits = 0
    for conn in connected:
        if isinstance(conn, str):
            if conn not in pending_ids:
                safe_exits += 1
        elif isinstance(conn, dict):
            if not conn.get("isDeathZone") and conn.get("id", "") not in pending_ids:
                safe_exits += 1
    f_safe_exits = _clamp(safe_exits / max(len(connected), 1))

    # ── Feature 6: enemy count (normalized) ─────────────────────────
    enemies = [a for a in visible_agents
               if isinstance(a, dict)
               and not a.get("isGuardian", False)
               and a.get("isAlive", True)
               and a.get("id") != self_data.get("id")]
    f_enemy_count = _clamp(len(enemies) / 10.0)

    # ── Feature 7: is nearest enemy weak? (their HP < my damage) ────
    from bot.strategy.brain import WEAPONS, calc_damage
    weapon_bonus = 0
    if equipped and isinstance(equipped, dict):
        weapon_bonus = WEAPONS.get(equipped.get("typeId", "").lower(), {}).get("bonus", 0)

    f_enemy_weak = 0.0
    if enemies:
        nearest = min(enemies, key=lambda a: a.get("hp", 999))
        my_dmg = calc_damage(atk, weapon_bonus, nearest.get("def", 5), weather)
        if nearest.get("hp", 100) <= my_dmg * 3:
            f_enemy_weak = 1.0

    # ── Feature 8: guardian nearby ───────────────────────────────────
    guardians_same = [a for a in visible_agents
                      if isinstance(a, dict)
                      and a.get("isGuardian", False)
                      and a.get("isAlive", True)
                      and a.get("regionId", "") == region_id]
    f_guardian_nearby = _clamp(len(guardians_same) / 3.0)

    # ── Feature 9: monster nearby ────────────────────────────────────
    monsters_here = [m for m in visible_monsters
                     if isinstance(m, dict) and m.get("hp", 0) > 0
                     and m.get("regionId", region_id) == region_id]
    f_monster_nearby = _clamp(len(monsters_here) / 5.0)

    # ── Feature 10: weapon bonus normalized ─────────────────────────
    f_weapon_bonus = _clamp(weapon_bonus / MAX_WEAPON_BONUS)

    # ── Feature 11: can act ──────────────────────────────────────────
    f_can_act = 1.0 if can_act else 0.0

    # ── Feature 12: heal item count ──────────────────────────────────
    heal_types = {"medkit", "bandage", "emergency_food"}
    heal_count = sum(1 for i in inventory
                     if isinstance(i, dict) and i.get("typeId", "").lower() in heal_types)
    f_heal_count = _clamp(heal_count / 5.0)

    # ── Feature 13: inventory space used ─────────────────────────────
    f_inv_used = _clamp(len(inventory) / 10.0)

    # ── Feature 14: has binoculars (vision boost) ────────────────────
    has_binos = any(isinstance(i, dict) and i.get("typeId", "").lower() == "binoculars"
                    for i in inventory)
    f_binos = 1.0 if has_binos else 0.0

    # ── Feature 15: terrain quality ──────────────────────────────────
    terrain_val = {"hills": 1.0, "plains": 0.8, "ruins": 0.6, "forest": 0.4, "water": 0.0}
    f_terrain = terrain_val.get(terrain, 0.5)

    # ── Feature 16: weather penalty ──────────────────────────────────
    weather_pen = {"clear": 1.0, "rain": 0.75, "fog": 0.5, "storm": 0.25}
    f_weather = weather_pen.get(weather, 0.75)

    # ── Feature 17: ATK relative to defense ─────────────────────────
    f_atk_ratio = _clamp((atk + weapon_bonus) / max(defense + 5, 1) / 4.0)

    # ── Feature 18: visible items in region ─────────────────────────
    visible_items = view.get("visibleItems", [])
    local_items = [i for i in visible_items
                   if isinstance(i, dict) and i.get("regionId", region_id) == region_id]
    f_items_nearby = _clamp(len(local_items) / 5.0)

    # ── Feature 19: has facility in region ──────────────────────────
    interactables = region.get("interactables", [])
    unused_facs = [f for f in interactables
                   if isinstance(f, dict) and not f.get("isUsed")]
    f_has_facility = _clamp(len(unused_facs) / 3.0)

    features = [
        f_hp,             # 0
        f_ep,             # 1
        f_alive_ratio,    # 2
        f_is_dz,          # 3
        f_pending_dz,     # 4
        f_safe_exits,     # 5
        f_enemy_count,    # 6
        f_enemy_weak,     # 7
        f_guardian_nearby, # 8
        f_monster_nearby, # 9
        f_weapon_bonus,   # 10
        f_can_act,        # 11
        f_heal_count,     # 12
        f_inv_used,       # 13
        f_binos,          # 14
        f_terrain,        # 15
        f_weather,        # 16
        f_atk_ratio,      # 17
        f_items_nearby,   # 18
        f_has_facility,   # 19
    ]
    return features


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))
