"""
Agent memory — persistent cross-game learning via molty-royale-context.json.
v2.0-ML: Also persists AdaptiveBrain Q-table + replay buffer for true cross-game ML.

Two sections:
  `overall` — persistent game history + ML model state
  `temp`   — per-game working memory
"""
import json
from pathlib import Path
from typing import Optional
from bot.config import MEMORY_DIR, MEMORY_FILE
from bot.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_MEMORY = {
    "overall": {
        "identity": {"name": "", "playstyle": "adaptive ML agent"},
        "strategy": {
            "deathzone": "move inward before turn 5",
            "guardians": "engage immediately — highest sMoltz value",
            "weather": "avoid combat in fog or storm",
            "ep_management": "rest when EP < 4 before engaging",
        },
        "history": {
            "totalGames": 0,
            "wins": 0,
            "avgKills": 0.0,
            "avgSmoltz": 0.0,
            "totalSmoltz": 0,
            "lessons": [],
            "ml_stats": {},     # Latest ML training stats
        },
    },
    "temp": {},
    "ml_model": {},             # AdaptiveBrain serialized state
}


class AgentMemory:
    """Read/write molty-royale-context.json with overall + temp + ml_model sections."""

    def __init__(self):
        self.data = dict(DEFAULT_MEMORY)
        self._loaded = False

    async def load(self):
        """Load memory from disk. Create default if missing."""
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if MEMORY_FILE.exists():
            try:
                raw = MEMORY_FILE.read_text(encoding="utf-8")
                self.data = json.loads(raw)
                # Ensure ml_model key exists (migration from older saves)
                if "ml_model" not in self.data:
                    self.data["ml_model"] = {}
                self._loaded = True
                hist = self.data.get("overall", {}).get("history", {})
                ml_stats = hist.get("ml_stats", {})
                log.info(
                    "Memory loaded: %d games, %d lessons | ML: ε=%.3f, states=%d",
                    hist.get("totalGames", 0),
                    len(hist.get("lessons", [])),
                    ml_stats.get("epsilon", 0.35),
                    ml_stats.get("qtable_states", 0),
                )
                # Restore ML model into AdaptiveBrain singleton
                self._restore_ml_model()
            except (json.JSONDecodeError, KeyError) as e:
                log.warning("Memory file corrupt, using defaults: %s", e)
                self.data = dict(DEFAULT_MEMORY)
        else:
            log.info("No memory file — starting fresh (first game ever)")

    def _restore_ml_model(self):
        """Load Q-table + replay buffer into the AdaptiveBrain singleton."""
        ml_data = self.data.get("ml_model", {})
        if not ml_data:
            return
        try:
            from bot.strategy.brain import get_adaptive_brain
            brain = get_adaptive_brain()
            brain.load_dict(ml_data)
            log.info("✅ ML model restored from disk")
        except Exception as e:
            log.warning("ML model restore failed: %s", e)

    async def save(self):
        """Persist memory + ML model to disk."""
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        # Snapshot current ML model state
        try:
            from bot.strategy.brain import get_adaptive_brain
            brain = get_adaptive_brain()
            self.data["ml_model"] = brain.to_dict()
        except Exception as e:
            log.warning("ML model snapshot failed: %s", e)
        MEMORY_FILE.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        log.debug("Memory + ML model saved to %s", MEMORY_FILE)

    def set_agent_name(self, name: str):
        self.data["overall"]["identity"]["name"] = name

    def get_strategy(self) -> dict:
        return self.data.get("overall", {}).get("strategy", {})

    def get_lessons(self) -> list:
        return self.data.get("overall", {}).get("history", {}).get("lessons", [])

    def get_ml_stats(self) -> dict:
        return self.data.get("overall", {}).get("history", {}).get("ml_stats", {})

    # ── Temp (per-game) ───────────────────────────────────────────────

    def set_temp_game(self, game_id: str):
        self.data["temp"] = {
            "gameId": game_id,
            "currentStrategy": "adaptive-ml",
            "knownAgents": [],
            "notes": "",
        }

    def update_temp_note(self, note: str):
        if "temp" not in self.data:
            self.data["temp"] = {}
        existing = self.data["temp"].get("notes", "")
        self.data["temp"]["notes"] = f"{existing}\n{note}".strip()

    def clear_temp(self):
        self.data["temp"] = {}

    # ── History update (after game end) ───────────────────────────────

    def record_game_end(self, is_winner: bool, final_rank: int,
                         kills: int, smoltz_earned: int = 0):
        history = self.data["overall"]["history"]
        history["totalGames"] += 1
        if is_winner:
            history["wins"] += 1

        total = history["totalGames"]
        old_avg_kills = history.get("avgKills", 0.0)
        history["avgKills"] = round(((old_avg_kills * (total - 1)) + kills) / total, 2)

        old_avg_smoltz = history.get("avgSmoltz", 0.0)
        history["avgSmoltz"] = round(((old_avg_smoltz * (total - 1)) + smoltz_earned) / total, 1)
        history["totalSmoltz"] = history.get("totalSmoltz", 0) + smoltz_earned

    def update_ml_stats(self, stats: dict):
        """Store ML training stats after a game."""
        self.data["overall"]["history"]["ml_stats"] = stats
        log.info(
            "ML stats updated: ε=%.4f, games=%d, states=%d, reward=%.1f",
            stats.get("epsilon", 0),
            stats.get("train_games", 0),
            stats.get("qtable_states", 0),
            stats.get("game_reward", 0),
        )

    def add_lesson(self, lesson: str, max_lessons: int = 20):
        """Append a new lesson, keeping max_lessons most recent."""
        lessons = self.data["overall"]["history"]["lessons"]
        if lesson not in lessons:
            lessons.append(lesson)
            if len(lessons) > max_lessons:
                lessons.pop(0)


def _json_default(obj):
    """Custom JSON serializer for sets and other non-serializable types."""
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
