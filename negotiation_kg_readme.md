# README: AI Negotiation Agent with Knowledge Graph



This project implements a **human-like AI negotiation agent** powered by a **Knowledge Graph (KG)**. The system engages in salary negotiations, tracks structured conversation history, and adapts its responses intelligently using a mix of:

- Prompt templates
- LangChain memory
- A dynamic Knowledge Graph (built using NetworkX)

The agent supports negotiation features such as:

- Memory of past offers (from both candidate and agent)
- Preference tracking (e.g., remote work)
- Adaptive concession strategies
- Offer rejection handling
- Final offer acceptance

---

 Key Components

 `negotiation_kg.py` — Knowledge Graph Backend

This module defines the **Knowledge Graph structure and logic**. It uses a directed graph to track:

- **Turns** in the conversation
- **Offers** (from candidate and agent)
- **Offer Statuses** (proposed, rejected, accepted)
- **Candidate Preferences** (e.g., perks like remote work)
- **Subjective Limits** (max salary offer allowed per turn)
- **Relations** like `SIMILAR_TO`, `REJECTED`, `PREFERS`

#### Key Methods:

- `add_turn()` → Logs a turn, including agent/candidate messages and salary ceiling.
- `add_offer()` → Stores structured offer details (base, perks, bonus) and who proposed it.
- `update_offer_status()` → Updates if an offer was rejected or accepted.
- `get_last_offer_details()` → Fetches the latest offer from either party.
- `get_offers_by_status()` → Lists all accepted/rejected offers.
- `add_candidate_preference()` → Captures and stores user preferences.
- `add_similar_offer_relation()` → Prevents repeat offers by marking them similar.
- `get_negotiation_summary()` → Generates a structured summary of all turns.

### 2. `negotiation_bot_kg.py` — Negotiation Agent Main Loop

This is the **main script** that runs the negotiation loop with user input. It connects the Knowledge Graph with the LLM.

#### Key Features:

- Uses LangChain with a ChatOpenAI-compatible model
- Employs a rich prompt template with embedded KG context
- Extracts structured offers from user and agent messages
- Dynamically adjusts the **subjective salary limit** per turn
- Detects rejections and acceptance using regex and updates the KG
- Injects candidate preferences and offer history into the prompt

---

## How It Works

###  Step-by-Step Negotiation Flow

1. **User Message:** Candidate proposes a salary or counter-offer
2. **Offer Parsing:** System extracts structured offer (base, perks, etc.)
3. **KG Update:**
   - Turn node is created
   - Offer node(s) are added and linked
   - If perks are mentioned, preference edges are created
4. **Subjective Limit Calculation:**
   - First turn → static value (e.g., \$115k)
   - Second turn → midpoint of last candidate offer and limit
   - One rejection → 8% bump
   - Multiple rejections → max limit (\$135k)
5. **Prompt Injection:** KG context is injected into the prompt (preferences, rejections, latest offers)
6. **Agent Response:** LLM generates a reply based on prompt + KG
7. **Response Parsing:** Extract agent offer, check similarity, update KG
8. **Finalization:** If candidate accepts, status is updated and conversation ends.

---

## Example KG Structure

```text
session_1
 └── HAS_TURN ──> turn_1
           └── CONTAINS_OFFER ──> offer_1_candidate (base: 130k)
           └── CONTAINS_OFFER ──> offer_1_agent (base: 115k)

candidate
 └── PREFERS ──> perk_remote_work
 └── REJECTED ──> offer_1_agent

turn_2
 └── CONTAINS_OFFER ──> offer_2_agent (base: 120k)
             └── SIMILAR_TO ──> offer_1_agent
```

---

## Highlights

- Avoids repeated offers using `SIMILAR_TO`
- Automatically finalizes when candidate says "deal"
- Tracks not just what was said, but what was **offered and preferred**
- Explainable and auditable via `get_negotiation_summary()`

---

##  How to Run

```bash
python negotiation_bot_kg.py
```

Interact with the agent via CLI. Type salary expectations and observe smart behavior.


