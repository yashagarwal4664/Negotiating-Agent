
# Adaptive Compensation Employer (ACE) Negotiation Agent

This project is an **adaptive, memory-aware employer negotiation agent** inspired by the [ACE: Adaptive Compensation Engine](https://arxiv.org/abs/2404.00911) paper. The system mimics human-like salary negotiation behavior by adjusting its salary ceiling dynamically based on the candidate's counteroffers while adhering to employer constraints.

---

## What This Version Adds (Compared to Static/Memoryless Agents)

- **Dynamic Ceiling Adjustment**:
  - Anchors low with a subjective limit (`$115,000`).
  - Concedes halfway on first counteroffer.
  - Reveals full budget ceiling (`$135,000`) on second or final turn.
- **Real-Time Memory via LangChain**:
  - Uses `ConversationBufferMemory` to recall the conversation history.
  - Avoids repeating offers or introductions.
- **Embedded Offer Extraction**:
  - Candidate salary asks are extracted via regex to adapt responses programmatically.

---

## Negotiation Flow Strategy

| Turn | Strategy                                                         |
|------|------------------------------------------------------------------|
| 0    | Low anchor offer (e.g., $115,000)                                |
| 1    | Concede halfway to the candidate's ask                           |
| ≥2   | Reveal full budget ceiling ($135,000) or finalize negotiation    |

---

##  Prerequisites

- Python 3.8+
- pip
- `.env` file containing:
  ```env
  OPENAI_API_KEY="your_openai_key"
  ```

The agent is configured to run using **UF-hosted LLaMA-3.1-70B** model.

---

## Setup Instructions

### 1. Clone and Create Environment

```bash
git clone https://github.com/your-username/ace-negotiation-agent.git
cd ace-negotiation-agent
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 2. Install Dependencies

```bash
pip install langchain langchain-openai python-dotenv
```

### 3. Add `.env`

Create a `.env` file with:

```env
OPENAI_API_KEY="your_api_key"
```

---

## Running the Agent

```bash
python ace_negotiation_agent.py
```

(Replace the filename with your script’s actual filename.)

You’ll see a prompt:

```
Negotiation Agent Active! Type your message as the candidate.
Type 'exit' to stop.
```

Interact with the employer agent and test different salary expectations.

---

## Sample Interaction

```
Candidate: I’m expecting $130,000
Employer Agent: Thanks for sharing that. While our initial budget is slightly lower, we're open to finding a middle ground...
Candidate: I need at least $132,000
Employer Agent: Understood. Let me see how close we can get—our best offer reaches $135,000 including stock and benefits...
```

---

## Files

| File                         | Description                                        |
|------------------------------|----------------------------------------------------|
| `ace_negotiation_agent.py`   | Main script implementing ACE-style negotiation     |
| `.env`                       | API key file                                       |
| `conversation.log`           | Log of dialogue for tracking & analysis            |

---

## Limitations

- Currently limited to numerical salary extraction only (may miss edge cases).
- Does not persist memory across sessions.
- Only supports base salary reasoning (perks/benefits are hardcoded).

---

## Inspired By

- **ACE: Adaptive Compensation Engine for AI-Driven Negotiation**  
  *[arXiv:2404.00911](https://arxiv.org/abs/2404.00911)*  
  Introduced principles for counteroffer pacing, anchoring, and emotional intelligence in salary negotiation agents.

---
