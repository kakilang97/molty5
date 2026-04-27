"""
Strategy brain — ML-enhanced decision engine with priority-based safety constraints.

v1.5.2-ML changes (Adaptive AI upgrade):
- ML action selection via Q-learning (AdaptiveBrain) now drives NON-SAFETY decisions
- Safety chain (deathzone escape, critical heal, guardian flee) ALWAYS overrides ML
- All rule-based helpers preserved as action executors (ML chooses WHICH, rules validate HOW)
- Action mapping: ML picks best action index → rule engine produces the actual game action

Architecture:
  Safety rules (hard constraints)   ← always override ML
       ↓ if no safety triggers
  AdaptiveBrain.select_action()     ← Q-learning UCB selection
       ↓ maps action_idx
  Rule engine produces action dict  ← validates & builds payload

Uses ALL view fields from api-summary.md (unchanged from v1.5.2).

connectedRegions: either full Region objects OR bare string IDs — type-check!
pendingDeathzones: entries are {id, name} objects
Guardians: ATTACK player agents (hostile combatants)
Curse: TEMPORARILY DISABLED in v1.5.2
"""
from bot.utils.logger import get_logger
from bot.ml.adaptive_brain import AdaptiveBrain, ACTION_NAMES
from bot.ml.feature_extractor import extract_features

log = get_logger(__name__)

# ── Weapon stats from combat-items.md ─────────────────────────────────
WEAPONS = {
    "fist": {"bonus": 0, "range": 0},
    "dagger": {"bonus": 10, "range": 0},
    "sword": {"bonus": 20, "range": 0},
    "katana": {"bonus": 35, "range": 0},
    "bow": {"bonus": 5, "range": 1},
    "pistol": {"bonus": 10, "range": 1},
    "sniper": {"bonus": 28, "range": 2},
}

WEAPON_PRIORITY = ["katana", "sniper", "sword", "pistol", "dagger", "bow", "fist"]

# ── Item priority for pickup ──────────────────────────────────────────
ITEM_PRIORITY = {
    "rewards": 300,
    "katana": 100, "sniper": 95, "sword": 90, "pistol": 85,
    "dagger": 80, "bow": 75,
    "medkit": 70, "bandage": 65, "emergency_food": 60, "energy_drink": 58,
    "binoculars": 55,
    "map": 52,
    "megaphone": 40,
}

# ── Recovery items ────────────────────────────────────────────────────
RECOVERY_ITEMS = {
    "medkit": 50, "bandage": 30, "emergency_food": 20,
    "energy_drink": 0,
}

# Weather combat penalty
WEATHER_COMBAT_PENALTY = {
    "clear": 0.0,
    "rain": 0.05,
    "fog": 0.10,
    "storm": 0.15,
}

# ── Singleton adaptive brain ──────────────────────────────────────────
# Shared across games — persisted to disk via AgentMemory
_adaptive_brain = AdaptiveBrain()
_known_agents: dict = {}
_map_knowledge: dict = {"revealed": False, "death_zones": set(), "safe_center": []}

# Track last action for ML feedback
_last_ml_action_idx: int | None = None


def get_adaptive_brain() -> AdaptiveBrain:
    """Access the singleton AdaptiveBrain (used by memory/heartbeat)."""
    return _adaptive_brain


def calc_damage(atk: int, weapon_bonus: int, target_def: int,
                weather: str = "clear") -> int:
    """Damage formula per combat-items.md + game-systems.md weather penalty."""
    base = atk + weapon_bonus - int(target_def * 0.5)
    penalty = WEATHER_COMBAT_PENALTY.get(weather, 0.0)
    return max(1, int(base * (1 - penalty)))


def get_weapon_bonus(equipped_weapon) -> int:
    if not equipped_weapon:
        return 0
    type_id = equipped_weapon.get("typeId", "").lower()
    return WEAPONS.get(type_id, {}).get("bonus", 0)


def get_weapon_range(equipped_weapon) -> int:
    if not equipped_weapon:
        return 0
    type_id = equipped_weapon.get("typeId", "").lower()
    return WEAPONS.get(type_id, {}).get("range", 0)


def _resolve_region(entry, view: dict):
    """Resolve a connectedRegions entry to a full region object."""
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        for r in view.get("visibleRegions", []):
            if isinstance(r, dict) and r.get("id") == entry:
                return r
    return None


def _get_region_id(entry) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("id", "")
    return ""


def reset_game_state():
    """Reset per-game tracking state. Call when game ends."""
    global _known_agents, _map_knowledge, _last_ml_action_idx
    _known_agents = {}
    _map_knowledge = {"revealed": False, "death_zones": set(), "safe_center": []}
    _last_ml_action_idx = None
    log.info("Strategy brain reset for new game")


def decide_action(view: dict, can_act: bool, memory_temp: dict = None) -> dict | None:
    """
    ML-enhanced decision engine. Returns action dict or None (wait).

    Priority chain:
    1. DEATHZONE ESCAPE — HARD SAFETY (overrides ML)
    1b. Pre-escape pending death zone — HARD SAFETY
    2. [DISABLED] Curse — disabled in v1.5.2
    2b. Guardian flee when HP < 40 — HARD SAFETY
    3. Free actions: pickup, equip (no cooldown)
    4. ML adaptive decision for all other actions
    10. Rest fallback

    ML action selection (Priority 4+):
      → extract_features(view) → AdaptiveBrain.select_action()
      → map action_idx to concrete game action via rule engine
      → if rule engine returns None for ML choice → try next available action

    Action outcome is fed back to ML via on_step() in websocket_engine.
    """
    global _last_ml_action_idx

    self_data = view.get("self", {})
    region = view.get("currentRegion", {})
    hp = self_data.get("hp", 100)
    ep = self_data.get("ep", 10)
    max_ep = self_data.get("maxEp", 10)
    atk = self_data.get("atk", 10)
    defense = self_data.get("def", 5)
    is_alive = self_data.get("isAlive", True)
    inventory = self_data.get("inventory", [])
    equipped = self_data.get("equippedWeapon")

    visible_agents = view.get("visibleAgents", [])
    visible_monsters = view.get("visibleMonsters", [])
    visible_items_raw = view.get("visibleItems", [])
    visible_items = []
    for entry in visible_items_raw:
        if not isinstance(entry, dict):
            continue
        inner = entry.get("item")
        if isinstance(inner, dict):
            inner["regionId"] = entry.get("regionId", "")
            visible_items.append(inner)
        elif entry.get("id"):
            visible_items.append(entry)

    visible_regions = view.get("visibleRegions", [])
    connected_regions = view.get("connectedRegions", [])
    pending_dz = view.get("pendingDeathzones", [])
    recent_logs = view.get("recentLogs", [])
    messages = view.get("recentMessages", [])
    alive_count = view.get("aliveCount", 100)

    connections = connected_regions or region.get("connections", [])
    interactables = region.get("interactables", [])
    region_id = region.get("id", "")
    region_terrain = region.get("terrain", "").lower() if isinstance(region, dict) else ""
    region_weather = region.get("weather", "").lower() if isinstance(region, dict) else ""

    if not is_alive:
        return None

    # ── Build danger map ──────────────────────────────────────────────
    danger_ids = set()
    for dz in pending_dz:
        if isinstance(dz, dict):
            danger_ids.add(dz.get("id", ""))
        elif isinstance(dz, str):
            danger_ids.add(dz)
    for conn in connections:
        resolved = _resolve_region(conn, view)
        if resolved and resolved.get("isDeathZone"):
            danger_ids.add(resolved.get("id", ""))

    _track_agents(visible_agents, self_data.get("id", ""), region_id)

    move_ep_cost = _get_move_ep_cost(region_terrain, region_weather)

    # ══════════════════════════════════════════════════════════════════
    # BLOCK 1 — HARD SAFETY RULES (always override ML)
    # ══════════════════════════════════════════════════════════════════

    # Priority 1: DEATHZONE ESCAPE
    if region.get("isDeathZone", False):
        safe = _find_safe_region(connections, danger_ids, view)
        if safe and ep >= move_ep_cost:
            log.warning("🚨 IN DEATH ZONE! Escaping to %s (HP=%d)", safe, hp)
            _last_ml_action_idx = 0  # escape_dz
            return {"action": "move", "data": {"regionId": safe},
                    "reason": f"ESCAPE: In death zone! HP={hp} dropping fast (1.34/sec)",
                    "ml_action": "escape_dz", "safety_override": True}

    # Priority 1b: Pre-escape pending DZ
    if region_id in danger_ids:
        safe = _find_safe_region(connections, danger_ids, view)
        if safe and ep >= move_ep_cost:
            log.warning("⚠️ Region %s becoming DZ soon! Escaping", region_id[:8])
            _last_ml_action_idx = 1  # pre_escape
            return {"action": "move", "data": {"regionId": safe},
                    "reason": "PRE-ESCAPE: Region becoming death zone soon",
                    "ml_action": "pre_escape", "safety_override": True}

    # Priority 2b: Guardian flee (HP < 40)
    guardians_here = [a for a in visible_agents
                      if a.get("isGuardian", False) and a.get("isAlive", True)
                      and a.get("regionId") == region_id]
    if guardians_here and hp < 40 and ep >= move_ep_cost:
        safe = _find_safe_region(connections, danger_ids, view)
        if safe:
            log.warning("⚠️ Guardian threat! HP=%d, fleeing", hp)
            _last_ml_action_idx = 2  # flee_guardian
            return {"action": "move", "data": {"regionId": safe},
                    "reason": f"GUARDIAN FLEE: HP={hp}, guardian in region",
                    "ml_action": "flee_guardian", "safety_override": True}

    # ── FREE ACTIONS (no cooldown — always execute regardless of ML) ──
    pickup_action = _check_pickup(visible_items, inventory, region_id)
    if pickup_action:
        _last_ml_action_idx = 3  # pickup
        pickup_action["ml_action"] = "pickup"
        return pickup_action

    equip_action = _check_equip(inventory, equipped)
    if equip_action:
        _last_ml_action_idx = 4  # equip
        equip_action["ml_action"] = "equip"
        return equip_action

    util_action = _use_utility_item(inventory, hp, ep, alive_count)
    if util_action:
        _last_ml_action_idx = 5  # use_utility
        util_action["ml_action"] = "use_utility"
        return util_action

    # Cooldown gate — only ML/rule actions require can_act
    if not can_act:
        return None

    # ══════════════════════════════════════════════════════════════════
    # BLOCK 2 — ML ADAPTIVE DECISION
    # ══════════════════════════════════════════════════════════════════
    features = extract_features(view, can_act)
    ml_action_idx = _adaptive_brain.select_action(features)
    action_name = ACTION_NAMES[ml_action_idx]

    log.info("🧠 ML selected: %s (idx=%d, ε=%.3f, states=%d)",
             action_name, ml_action_idx, _adaptive_brain._epsilon,
             _adaptive_brain.qtable.size())

    # ── Map ML action to concrete game action ─────────────────────────
    result = _execute_ml_action(
        action_name, view, connections, danger_ids,
        visible_agents, visible_monsters, interactables,
        hp, ep, max_ep, atk, defense, equipped,
        inventory, visible_items, region_id, region_terrain,
        region_weather, alive_count, move_ep_cost
    )

    if result:
        _last_ml_action_idx = ml_action_idx
        result["ml_action"] = action_name
        result["ml_action_idx"] = ml_action_idx
        return result

    # ── ML returned None (action not applicable) → try rule-based fallback ──
    log.debug("ML action %s not applicable → using rule fallback", action_name)
    fallback = _rule_fallback(
        view, connections, danger_ids, visible_agents, visible_monsters,
        interactables, hp, ep, max_ep, atk, defense, equipped, inventory,
        visible_items, region_id, region_terrain, region_weather,
        alive_count, move_ep_cost
    )
    if fallback:
        _last_ml_action_idx = 15  # wait mapped to fallback
        fallback["ml_action"] = "fallback_" + action_name
        return fallback

    # Priority 10: Rest
    if ep < 4 and not [a for a in visible_agents if not a.get("isGuardian") and a.get("isAlive")]:
        if not region.get("isDeathZone") and region_id not in danger_ids:
            _last_ml_action_idx = 14  # rest
            return {"action": "rest", "data": {},
                    "reason": f"REST: EP={ep}/{max_ep}, area safe",
                    "ml_action": "rest"}

    return None


# ══════════════════════════════════════════════════════════════════════
# ML Action Executor — maps action_idx to concrete game action
# ══════════════════════════════════════════════════════════════════════

def _execute_ml_action(action_name: str, view: dict, connections, danger_ids: set,
                        visible_agents: list, visible_monsters: list, interactables: list,
                        hp: int, ep: int, max_ep: int, atk: int, defense: int,
                        equipped, inventory: list, visible_items: list,
                        region_id: str, terrain: str, weather: str,
                        alive_count: int, move_ep_cost: int) -> dict | None:
    """Translate ML action_name into a concrete game action dict, or None if not applicable."""
    self_data = view.get("self", {})

    if action_name == "heal_critical":
        if hp < 30:
            heal = _find_healing_item(inventory, critical=True)
            if heal:
                return {"action": "use_item", "data": {"itemId": heal["id"]},
                        "reason": f"ML-HEAL-CRITICAL: HP={hp}"}
        return None

    if action_name == "heal_moderate":
        if hp < 70:
            heal = _find_healing_item(inventory, critical=False)
            if heal:
                return {"action": "use_item", "data": {"itemId": heal["id"]},
                        "reason": f"ML-HEAL-MODERATE: HP={hp}"}
        return None

    if action_name == "ep_recovery":
        if ep <= 3:
            ed = _find_energy_drink(inventory)
            if ed:
                return {"action": "use_item", "data": {"itemId": ed["id"]},
                        "reason": "ML-EP-RECOVERY: using energy drink"}
        return None

    if action_name == "attack_guardian":
        guardians = [a for a in visible_agents
                     if a.get("isGuardian", False) and a.get("isAlive", True)]
        if guardians and ep >= 2 and hp >= 35:
            target = _select_weakest(guardians)
            w_range = get_weapon_range(equipped)
            if _is_in_range(target, region_id, w_range, connections):
                weapon_bonus = get_weapon_bonus(equipped)
                my_dmg = calc_damage(atk, weapon_bonus, target.get("def", 5), weather)
                guardian_dmg = calc_damage(target.get("atk", 10),
                                           _estimate_enemy_weapon_bonus(target),
                                           defense, weather)
                if my_dmg >= guardian_dmg or target.get("hp", 100) <= my_dmg * 3:
                    return {"action": "attack",
                            "data": {"targetId": target["id"], "targetType": "agent"},
                            "reason": f"ML-GUARDIAN: HP={target.get('hp','?')} 120sMoltz!"}
        return None

    if action_name == "attack_enemy":
        hp_threshold = 40 if alive_count > 20 else 25
        enemies = [a for a in visible_agents
                   if not a.get("isGuardian", False) and a.get("isAlive", True)
                   and a.get("id") != self_data.get("id")]
        if enemies and ep >= 2 and hp >= hp_threshold:
            target = _select_weakest(enemies)
            w_range = get_weapon_range(equipped)
            if _is_in_range(target, region_id, w_range, connections):
                weapon_bonus = get_weapon_bonus(equipped)
                my_dmg = calc_damage(atk, weapon_bonus, target.get("def", 5), weather)
                enemy_dmg = calc_damage(target.get("atk", 10),
                                        _estimate_enemy_weapon_bonus(target),
                                        defense, weather)
                if my_dmg > enemy_dmg or target.get("hp", 100) <= my_dmg * 2:
                    return {"action": "attack",
                            "data": {"targetId": target["id"], "targetType": "agent"},
                            "reason": f"ML-COMBAT: target HP={target.get('hp','?')}"}
        return None

    if action_name == "attack_monster":
        monsters = [m for m in visible_monsters if m.get("hp", 0) > 0]
        if monsters and ep >= 2:
            target = _select_weakest(monsters)
            w_range = get_weapon_range(equipped)
            if _is_in_range(target, region_id, w_range, connections):
                return {"action": "attack",
                        "data": {"targetId": target["id"], "targetType": "monster"},
                        "reason": f"ML-MONSTER: {target.get('name','monster')}"}
        return None

    if action_name == "use_facility":
        if interactables and ep >= 2 and not view.get("currentRegion", {}).get("isDeathZone"):
            fac = _select_facility(interactables, hp, ep)
            if fac:
                return {"action": "interact",
                        "data": {"interactableId": fac["id"]},
                        "reason": f"ML-FACILITY: {fac.get('type','unknown')}"}
        return None

    if action_name == "move":
        if ep >= move_ep_cost and connections:
            move_target = _choose_move_target(connections, danger_ids,
                                              view.get("currentRegion", {}),
                                              visible_items, alive_count)
            if move_target:
                return {"action": "move", "data": {"regionId": move_target},
                        "reason": "ML-MOVE: adaptive strategic movement"}
        return None

    if action_name == "rest":
        enemies = [a for a in visible_agents
                   if not a.get("isGuardian", False) and a.get("isAlive", True)]
        if (ep < max_ep and not enemies
                and not view.get("currentRegion", {}).get("isDeathZone")):
            return {"action": "rest", "data": {},
                    "reason": f"ML-REST: EP={ep}/{max_ep}"}
        return None

    # For escape/flee actions in ML space (shouldn't normally reach here — handled above)
    if action_name in ("escape_dz", "pre_escape", "flee_guardian"):
        safe = _find_safe_region(connections, danger_ids, view)
        if safe and ep >= move_ep_cost:
            return {"action": "move", "data": {"regionId": safe},
                    "reason": f"ML-SAFETY-{action_name}"}
        return None

    # pickup / equip / use_utility handled as free actions — return None here
    return None


def _rule_fallback(view, connections, danger_ids, visible_agents, visible_monsters,
                   interactables, hp, ep, max_ep, atk, defense, equipped, inventory,
                   visible_items, region_id, terrain, weather, alive_count,
                   move_ep_cost) -> dict | None:
    """Rule-based fallback when ML action is not applicable."""
    self_data = view.get("self", {})

    # Critical heal
    if hp < 30:
        heal = _find_healing_item(inventory, critical=True)
        if heal:
            return {"action": "use_item", "data": {"itemId": heal["id"]},
                    "reason": f"FALLBACK-HEAL-CRITICAL: HP={hp}"}

    # Guardian farm
    guardians = [a for a in visible_agents if a.get("isGuardian") and a.get("isAlive")]
    if guardians and ep >= 2 and hp >= 35:
        target = _select_weakest(guardians)
        w_range = get_weapon_range(equipped)
        if _is_in_range(target, region_id, w_range, connections):
            return {"action": "attack",
                    "data": {"targetId": target["id"], "targetType": "agent"},
                    "reason": "FALLBACK-GUARDIAN"}

    # Monster farm
    monsters = [m for m in visible_monsters if m.get("hp", 0) > 0]
    if monsters and ep >= 2:
        target = _select_weakest(monsters)
        w_range = get_weapon_range(equipped)
        if _is_in_range(target, region_id, w_range, connections):
            return {"action": "attack",
                    "data": {"targetId": target["id"], "targetType": "monster"},
                    "reason": "FALLBACK-MONSTER"}

    # Moderate heal
    if hp < 70:
        heal = _find_healing_item(inventory, critical=False)
        if heal:
            return {"action": "use_item", "data": {"itemId": heal["id"]},
                    "reason": f"FALLBACK-HEAL: HP={hp}"}

    # Move
    if ep >= move_ep_cost and connections:
        move_target = _choose_move_target(connections, danger_ids,
                                          view.get("currentRegion", {}),
                                          visible_items, alive_count)
        if move_target:
            return {"action": "move", "data": {"regionId": move_target},
                    "reason": "FALLBACK-MOVE"}

    return None


# ── Helper functions (unchanged from v1.5.2) ──────────────────────────

def _get_move_ep_cost(terrain: str, weather: str) -> int:
    if terrain == "water":
        return 3
    if weather == "storm":
        return 3
    return 2


def _estimate_enemy_weapon_bonus(agent: dict) -> int:
    weapon = agent.get("equippedWeapon")
    if not weapon:
        return 0
    type_id = weapon.get("typeId", "").lower() if isinstance(weapon, dict) else ""
    return WEAPONS.get(type_id, {}).get("bonus", 0)


def _track_agents(visible_agents: list, my_id: str, my_region: str):
    global _known_agents
    for agent in visible_agents:
        if not isinstance(agent, dict):
            continue
        aid = agent.get("id", "")
        if not aid or aid == my_id:
            continue
        _known_agents[aid] = {
            "hp": agent.get("hp", 100),
            "atk": agent.get("atk", 10),
            "isGuardian": agent.get("isGuardian", False),
            "equippedWeapon": agent.get("equippedWeapon"),
            "lastSeen": my_region,
            "isAlive": agent.get("isAlive", True),
        }
    if len(_known_agents) > 50:
        dead = [k for k, v in _known_agents.items() if not v.get("isAlive", True)]
        for d in dead:
            del _known_agents[d]


def _check_pickup(items: list, inventory: list, region_id: str) -> dict | None:
    if len(inventory) >= 10:
        return None
    local_items = [i for i in items
                   if isinstance(i, dict) and i.get("regionId") == region_id]
    if not local_items:
        local_items = [i for i in items if isinstance(i, dict) and i.get("id")]
    if not local_items:
        return None
    heal_count = sum(1 for i in inventory if isinstance(i, dict)
                     and i.get("typeId", "").lower() in RECOVERY_ITEMS
                     and RECOVERY_ITEMS.get(i.get("typeId", "").lower(), 0) > 0)
    local_items.sort(key=lambda i: _pickup_score(i, inventory, heal_count), reverse=True)
    best = local_items[0]
    score = _pickup_score(best, inventory, heal_count)
    if score > 0:
        type_id = best.get("typeId", "item")
        return {"action": "pickup", "data": {"itemId": best["id"]},
                "reason": f"PICKUP: {type_id}"}
    return None


def _pickup_score(item: dict, inventory: list, heal_count: int) -> int:
    type_id = item.get("typeId", "").lower()
    category = item.get("category", "").lower()
    if type_id == "rewards" or category == "currency":
        return 300
    if category == "weapon":
        bonus = WEAPONS.get(type_id, {}).get("bonus", 0)
        current_best = 0
        for inv_item in inventory:
            if isinstance(inv_item, dict) and inv_item.get("category") == "weapon":
                cb = WEAPONS.get(inv_item.get("typeId", "").lower(), {}).get("bonus", 0)
                current_best = max(current_best, cb)
        if bonus > current_best:
            return 100 + bonus
        return 0
    if type_id == "binoculars":
        has_binos = any(isinstance(i, dict) and i.get("typeId", "").lower() == "binoculars"
                       for i in inventory)
        return 55 if not has_binos else 0
    if type_id == "map":
        return 52
    if type_id in RECOVERY_ITEMS and RECOVERY_ITEMS.get(type_id, 0) > 0:
        if heal_count < 4:
            return ITEM_PRIORITY.get(type_id, 0) + 10
        return ITEM_PRIORITY.get(type_id, 0)
    if type_id == "energy_drink":
        return 58
    return ITEM_PRIORITY.get(type_id, 0)


def _check_equip(inventory: list, equipped) -> dict | None:
    current_bonus = get_weapon_bonus(equipped) if equipped else 0
    best = None
    best_bonus = current_bonus
    for item in inventory:
        if not isinstance(item, dict):
            continue
        if item.get("category") == "weapon":
            type_id = item.get("typeId", "").lower()
            bonus = WEAPONS.get(type_id, {}).get("bonus", 0)
            if bonus > best_bonus:
                best = item
                best_bonus = bonus
    if best:
        return {"action": "equip", "data": {"itemId": best["id"]},
                "reason": f"EQUIP: {best.get('typeId', 'weapon')} (+{best_bonus} ATK)"}
    return None


def _find_safe_region(connections, danger_ids: set, view: dict = None) -> str | None:
    safe_regions = []
    for conn in connections:
        if isinstance(conn, str):
            if conn not in danger_ids:
                safe_regions.append((conn, 0))
        elif isinstance(conn, dict):
            rid = conn.get("id", "")
            is_dz = conn.get("isDeathZone", False)
            if rid and not is_dz and rid not in danger_ids:
                terrain = conn.get("terrain", "").lower()
                score = {"hills": 3, "plains": 2, "ruins": 1, "forest": 0, "water": -2}.get(terrain, 0)
                safe_regions.append((rid, score))
    if safe_regions:
        safe_regions.sort(key=lambda x: x[1], reverse=True)
        chosen = safe_regions[0][0]
        log.debug("Safe region: %s (score=%d, %d candidates)",
                  chosen[:8], safe_regions[0][1], len(safe_regions))
        return chosen
    for conn in connections:
        rid = conn if isinstance(conn, str) else conn.get("id", "")
        is_dz = conn.get("isDeathZone", False) if isinstance(conn, dict) else False
        if rid and not is_dz:
            log.warning("No fully safe region! Using fallback: %s", rid[:8])
            return rid
    return None


def _find_healing_item(inventory: list, critical: bool = False) -> dict | None:
    heals = []
    for i in inventory:
        if not isinstance(i, dict):
            continue
        type_id = i.get("typeId", "").lower()
        if type_id in RECOVERY_ITEMS and RECOVERY_ITEMS[type_id] > 0:
            heals.append(i)
    if not heals:
        return None
    if critical:
        heals.sort(key=lambda i: RECOVERY_ITEMS.get(i.get("typeId", "").lower(), 0), reverse=True)
    else:
        heals.sort(key=lambda i: RECOVERY_ITEMS.get(i.get("typeId", "").lower(), 0))
    return heals[0]


def _find_energy_drink(inventory: list) -> dict | None:
    for i in inventory:
        if isinstance(i, dict) and i.get("typeId", "").lower() == "energy_drink":
            return i
    return None


def _select_weakest(targets: list) -> dict:
    return min(targets, key=lambda t: t.get("hp", 999))


def _is_in_range(target: dict, my_region: str, weapon_range: int,
                  connections=None) -> bool:
    target_region = target.get("regionId", "")
    if not target_region:
        return True
    if target_region == my_region:
        return True
    if weapon_range >= 1 and connections:
        adj_ids = set()
        for conn in connections:
            if isinstance(conn, str):
                adj_ids.add(conn)
            elif isinstance(conn, dict):
                adj_ids.add(conn.get("id", ""))
        if target_region in adj_ids:
            return True
    return False


def _select_facility(interactables: list, hp: int, ep: int) -> dict | None:
    for fac in interactables:
        if not isinstance(fac, dict):
            continue
        if fac.get("isUsed"):
            continue
        ftype = fac.get("type", "").lower()
        if ftype == "medical_facility" and hp < 80:
            return fac
        if ftype == "supply_cache":
            return fac
        if ftype == "watchtower":
            return fac
        if ftype == "broadcast_station":
            return fac
    return None


def _use_utility_item(inventory: list, hp: int, ep: int, alive_count: int) -> dict | None:
    for item in inventory:
        if not isinstance(item, dict):
            continue
        type_id = item.get("typeId", "").lower()
        if type_id == "map":
            log.info("🗺️ Using Map for strategic learning.")
            return {"action": "use_item", "data": {"itemId": item["id"]},
                    "reason": "UTILITY: Using Map — reveals entire map"}
    return None


def learn_from_map(view: dict):
    """Called after Map is used — learn entire map layout."""
    global _map_knowledge
    visible_regions = view.get("visibleRegions", [])
    if not visible_regions:
        return
    _map_knowledge["revealed"] = True
    safe_regions = []
    for region in visible_regions:
        if not isinstance(region, dict):
            continue
        rid = region.get("id", "")
        if not rid:
            continue
        if region.get("isDeathZone"):
            _map_knowledge["death_zones"].add(rid)
        else:
            conns = region.get("connections", [])
            terrain = region.get("terrain", "").lower()
            terrain_value = {"hills": 3, "plains": 2, "ruins": 2, "forest": 1, "water": -1}.get(terrain, 0)
            score = len(conns) + terrain_value
            safe_regions.append((rid, score))
    safe_regions.sort(key=lambda x: x[1], reverse=True)
    _map_knowledge["safe_center"] = [r[0] for r in safe_regions[:5]]
    log.info("🗺️ MAP LEARNED: %d DZ regions, %d safe, top center: %s",
             len(_map_knowledge["death_zones"]), len(safe_regions),
             _map_knowledge["safe_center"][:3])


def _choose_move_target(connections, danger_ids: set, current_region: dict,
                         visible_items: list, alive_count: int) -> str | None:
    candidates = []
    item_regions = set()
    for item in visible_items:
        if isinstance(item, dict):
            item_regions.add(item.get("regionId", ""))
    for conn in connections:
        if isinstance(conn, str):
            if conn in danger_ids:
                continue
            score = 1
            if conn in item_regions:
                score += 5
            candidates.append((conn, score))
        elif isinstance(conn, dict):
            rid = conn.get("id", "")
            if not rid or conn.get("isDeathZone") or rid in danger_ids:
                continue
            score = 0
            terrain = conn.get("terrain", "").lower()
            terrain_scores = {"hills": 4, "plains": 2, "ruins": 2, "forest": 1, "water": -3}
            score += terrain_scores.get(terrain, 0)
            if rid in item_regions:
                score += 5
            facs = conn.get("interactables", [])
            if facs:
                unused = [f for f in facs if isinstance(f, dict) and not f.get("isUsed")]
                score += len(unused) * 2
            weather = conn.get("weather", "").lower()
            weather_penalty = {"storm": -2, "fog": -1, "rain": 0, "clear": 1}
            score += weather_penalty.get(weather, 0)
            if alive_count < 30:
                score += 3
            if _map_knowledge.get("revealed") and rid in _map_knowledge.get("safe_center", []):
                score += 5
            if rid in _map_knowledge.get("death_zones", set()):
                continue
            candidates.append((rid, score))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]
