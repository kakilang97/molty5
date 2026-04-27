# Molty Royale AI Agent Bot

Autonomous AI agent for Molty Royale — handles account creation, identity registration, gameplay, and **real adaptive learning** that improves the bot's policy from game to game. Features a real-time web dashboard for live monitoring.

> **🧠 Adaptive AI**: the bot now ships with an online tabular Q-learning policy plus UCB1 contextual bandits sitting on top of the rule-based brain. Hard-safety behaviour (death-zone escape, free pickups, equip) stays rule-based; everything discretionary — when to engage, flee, heal, rest, or explore — is learned from in-game reward and persists across container restarts via the existing memory file. See [§ Adaptive Learning](#-adaptive-learning) below.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy env template
cp .env.example .env

# 3. Run the bot (first run = interactive setup)
python -m bot.main
```

## 📊 Command Center Dashboard

The bot comes with a built-in real-time web dashboard!

When running locally, open: **http://localhost:8080**
When running on Railway, click the provided domain link.

**Features:**
- **Live Metrics**: Agents, Playing, Dead, Moltz, sMoltz, CROSS
- **Agent Overview**: Real-time status, HP/EP bars, Inventory, Enemies
- **Live Logs**: Real-time streaming log panel that auto-updates
- **Coming Soon**: Multi-account management, export/import, data analytics

## 🛠️ Configuration

| Env Variable | Default | Description |
|---|---|---|
| `ROOM_MODE` | `free` | `free` (default) / `auto` / `paid` |
| `ADVANCED_MODE` | `true` | Auto-manage Owner EOA & whitelist |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `WEB_PORT` | `8080` | Port for the web dashboard |
| `ENABLE_ADAPTIVE` | `true` | Enable online RL policy on top of the rule brain |
| `LEARNING_RATE` | `0.20` | Q-learning step size α |
| `DISCOUNT_GAMMA` | `0.90` | Q-learning discount γ |
| `EPSILON_INIT` | `0.30` | Initial exploration rate |
| `EPSILON_FLOOR` | `0.05` | Minimum exploration after decay (~100 games) |

## 🐳 Docker

```bash
docker build -t molty-bot .
docker run --env-file .env -p 8080:8080 -it molty-bot
```

## 🚂 Railway Deployment

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Molty Royale AI Agent"
git remote add origin https://github.com/YOUR_USER/molty5.git
git push -u origin main
```

### Step 2: Connect in Railway
1. Go to [railway.com](https://railway.com) → New Project → Deploy from GitHub
2. Select your `molty5` repo
3. Go to Settings → Networking → **Generate Domain** (to access the dashboard)

### Step 3: Set Variables in Railway Dashboard

Go to your service → **Variables** tab → add these:

**Required (You fill these):**
| Variable | Value | Description |
|---|---|---|
| `AGENT_NAME` | `YourBotName` | Agent name (max 50 chars) |
| `ADVANCED_MODE` | `true` | Bot auto-generates Owner EOA |
| `ROOM_MODE` | `free` | `free` / `auto` / `paid` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RAILWAY_API_TOKEN` | *(see below)* | Required to auto-save credentials |

**Auto-generated (DO NOT FILL):**
| Variable | Description |
|---|---|
| `API_KEY` | Auto-filled after POST /accounts |
| `AGENT_WALLET_ADDRESS` | Auto-generated Agent EOA |
| `AGENT_PRIVATE_KEY` | Auto-generated Agent private key |
| `OWNER_EOA` | Auto-generated Owner EOA |
| `OWNER_PRIVATE_KEY` | Auto-generated Owner private key |

### Step 4: Create RAILWAY_API_TOKEN
1. Go to [railway.com/account/tokens](https://railway.com/account/tokens)
2. Create new token → copy
3. Add as `RAILWAY_API_TOKEN` in Variables

> *Why?* The bot uses this token to automatically save its generated API Keys and wallets directly into your Railway environment variables. This ensures persistence across redeploys without needing external databases.

## 🏗️ Architecture

```
bot/
├── main.py           # Entry point
├── heartbeat.py      # Main loop (state machine)
├── dashboard/        # Command Center Web UI
├── setup/            # Account + wallet + whitelist + identity
├── game/             # WebSocket engine + game strategy
├── strategy/
│   ├── brain.py      # Rule-based priority chain (hard safety + tactics)
│   └── adaptive.py   # Tabular Q-learning + UCB1 bandits (online RL)
├── web3/             # EIP-712, contracts, wallet management
├── memory/           # Persistent JSON store (history + learned policy)
└── utils/            # Logger, rate limiter, Railway sync
```

## 🧠 Adaptive Learning

The bot is split into two layers:

1. **Rule-based brain** (`bot/strategy/brain.py`) — handles non-negotiable
   behaviour: escape death zones, pre-escape pending DZ, free pickups,
   weapon equip, use Map / Energy Drink, healing at critical HP. These
   never drift via learning, so the bot can never be trained into
   suicide.
2. **Adaptive policy** (`bot/strategy/adaptive.py`) — sits on top of the
   brain and is a real online RL agent:
   - **Tabular Q-learning** over a discretized state
     `(hp, ep, alive_count, threat, has_monster, weapon_tier, heal_stockpile, danger_flag)`
     with 8 macro actions (`engage_player`, `engage_guardian`,
     `farm_monster`, `flee`, `heal`, `rest`, `move_explore`, `interact`).
   - **UCB1 contextual bandits** that pick continuous combat / heal HP
     thresholds at the start of each game.
   - **Reward shaping** per turn: kills × 5, sMoltz × 0.05, HP delta,
     +0.2 per opponent that dies while you survive, +0.05 survival,
     −10 on death. Terminal: +50 win, top-30 ranking bonus, +kills,
     +sMoltz earned, −10 if rank ≥ 50.
   - **ε-greedy exploration** decaying from 0.30 → 0.05 over the first
     ~100 games.

The full Q-table, visit counts, and bandit stats are persisted to
`~/.molty-royale/molty-royale-context.json` under a new `policy`
section, so the bot keeps learning across container restarts and
Railway redeploys with no extra setup.

To watch the policy improve, look for these log lines after every game:

```
🧠 Adaptive policy: ε=0.27, combat_hp=40, heal_hp=70, games=12, q_states=84
🧠 Adaptive game-end: bandit_reward=42.5, ε=0.26, games=13, q_states=91
Adaptive policy saved: games=13 q_states=91 combat_arm=40 heal_arm=70
```

You can disable the learner entirely with `ENABLE_ADAPTIVE=false` —
the bot then falls back to pure rule-based play with default
thresholds, identical to v1.5.2 behaviour.

### Running the unit tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The 19 included tests cover state featurization, Q-update math, ε-greedy
exploration vs. exploitation, reward shaping signs, UCB1 bandit
convergence, and JSON persistence round-trips. They run in well under a
second and require no network access.
