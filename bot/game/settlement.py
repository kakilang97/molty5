"""
Game settlement — Phase 3: process game end, update memory, prepare for next game.
"""
from bot.memory.agent_memory import AgentMemory
from bot.strategy.adaptive import AdaptiveLearner
from bot.utils.logger import get_logger

log = get_logger(__name__)


async def settle_game(
    game_result: dict,
    entry_type: str,
    memory: AgentMemory,
    learner: AdaptiveLearner | None = None,
):
    """
    Process game end:
    1. Extract final stats
    2. Update memory (overall history + lessons)
    3. Persist adaptive policy (Q-table + bandits) when a learner is given
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

    # Update memory
    memory.record_game_end(
        is_winner=is_winner,
        final_rank=final_rank,
        kills=kills,
        smoltz_earned=smoltz_earned,
    )

    # Add lessons based on game outcome
    if is_winner:
        memory.add_lesson(f"Won with {kills} kills at rank {final_rank}")
    elif final_rank <= 3:
        memory.add_lesson(f"Top 3 finish (rank {final_rank}) with {kills} kills")
    elif kills == 0:
        memory.add_lesson("Zero kills — need more aggressive guardian/monster targeting")

    # Persist adaptive policy (Q-table + bandit stats) so learning carries
    # over across games and container restarts.
    if learner is not None:
        memory.set_policy(learner.to_dict())
        log.info(
            "Adaptive policy saved: games=%d q_states=%d combat_arm=%d heal_arm=%d",
            learner.games_played, len(learner.q_table),
            learner.last_combat_arm, learner.last_heal_arm,
        )

    # Clear temp for next game
    memory.clear_temp()
    await memory.save()

    log.info("Settlement complete. Ready for next game.")
