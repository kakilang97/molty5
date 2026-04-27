"""
Game settlement — Phase 3: process game end, update memory, prepare for next game.
v2.0-ML: Also persists AdaptiveBrain training stats into memory.
"""
from bot.memory.agent_memory import AgentMemory
from bot.dashboard.state import dashboard_state
from bot.utils.logger import get_logger

log = get_logger(__name__)


async def settle_game(game_result: dict, entry_type: str, memory: AgentMemory):
    """
    Process game end:
    1. Extract final stats
    2. Update memory (overall history + lessons)
    3. Persist ML model stats
    4. Clear temp memory
    """
    result = game_result.get("result", game_result)
    is_winner = result.get("isWinner", False)
    final_rank = result.get("finalRank", 0)
    kills = result.get("kills", 0)
    rewards = result.get("rewards", {})
    smoltz_earned = rewards.get("sMoltz", 0)
    moltz_earned = rewards.get("moltz", 0)

    log.info("═══ GAME SETTLEMENT ═══")
    log.info("  Winner: %s | Rank: %d | Kills: %d", "YES" if is_winner else "No", final_rank, kills)
    log.info("  Rewards: %d sMoltz, %d Moltz", smoltz_earned, moltz_earned)

    # Update game history
    memory.record_game_end(
        is_winner=is_winner,
        final_rank=final_rank,
        kills=kills,
        smoltz_earned=smoltz_earned,
    )

    # Persist ML training stats collected during the game
    ml_stats = getattr(dashboard_state, "ml_stats", None)
    if ml_stats:
        memory.update_ml_stats(ml_stats)
        log.info("🧠 ML stats persisted: ε=%.4f, states=%d",
                 ml_stats.get("epsilon", 0),
                 ml_stats.get("qtable_states", 0))
        dashboard_state.ml_stats = None

    # Add lessons based on game outcome
    if is_winner:
        memory.add_lesson(f"Won with {kills} kills at rank {final_rank}")
    elif final_rank <= 3:
        memory.add_lesson(f"Top 3 finish (rank {final_rank}) with {kills} kills")
    elif kills == 0:
        memory.add_lesson("Zero kills — need more aggressive guardian/monster targeting")

    if ml_stats:
        game_reward = ml_stats.get("game_reward", 0)
        if game_reward > 50:
            memory.add_lesson(f"High ML reward game ({game_reward:.0f}) — profitable strategy learned")
        elif game_reward < -15:
            memory.add_lesson(f"Low ML reward ({game_reward:.0f}) — death zone or poor combat choices")

    # Clear temp for next game
    memory.clear_temp()
    await memory.save()

    log.info("Settlement complete. Ready for next game.")
