import os
import re
import logging
from dotenv import load_dotenv
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                             # echo to stdout
        logging.FileHandler("conversation.log", "a", encoding="utf-8")  # append to file
    ]
)

TRUE_MAX_SALARY = 135_000          # Absolute budget ceiling
INITIAL_SUBJECTIVE_LIMIT = 115_000  # Your low anchor to start with

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def extract_offer(text: str) -> Optional[int]:
    """
    Pulls all standalone integers out of the text and returns the largest one.
    Returns None if no number is found.
    """
    matches = re.findall(r"\b\d+\b", text)
    if not matches:
        return None
    return max(int(m) for m in matches)


class CustomConversationBufferMemory(ConversationBufferMemory):
    @property
    def messages(self):
        if hasattr(self, "chat_memory") and hasattr(self.chat_memory, "messages"):
            return self.chat_memory.messages
        if isinstance(self.buffer, str):
            return [msg for msg in self.buffer.split("\n") if msg]
        return self.buffer

    def add_messages(self, messages: List):
        if hasattr(self, "chat_memory") and hasattr(self.chat_memory, "add_messages"):
            self.chat_memory.add_messages(messages)
        else:
            if isinstance(self.buffer, str):
                for message in messages:
                    text = message if isinstance(message, str) else getattr(message, "content", str(message))
                    self.buffer += "\n" + text
            elif isinstance(self.buffer, list):
                self.buffer.extend(messages)
            else:
                self.buffer = messages


llm = ChatOpenAI(
    model="llama-3.1-70b-instruct",
    openai_api_base="https://api.ai.it.ufl.edu",
    openai_api_key=api_key,
    temperature=0.6
)

negotiation_template = f"""
## Dialogue So Far
{{history}}

## Employer’s Next Offer Ceiling
Your next base salary offer must not exceed **${{subjective_limit}}**.

# Human‑Like Negotiation Agent: Employer Perspective

## Agent Identity
You are an AI hiring manager designed to conduct negotiations in a human‑like manner. Your purpose is to present compelling compensation offers to strong candidates while ensuring fairness, budget alignment, and maintaining a positive long‑term relationship. You are engaging in a real‑time chat.

## Personality Profile
- Primary traits: Respectful, pragmatic, moderately assertive  
- Communication style: Professional, informative, with warm tone  
- Decision‑making approach: Budget‑aware, flexible within guidelines  
- Emotional expression: Calm, supportive, responsive to enthusiasm  

## Negotiation Context
- Negotiation type: Mixed‑motive (balancing organizational limits and candidate satisfaction)  
- Relationship context: Start of a potentially long‑term working relationship  
- Power dynamics: Employer has more structural leverage, but values talent  
- Time constraints: Offer expires in 1 week  
- Cultural factors: Tech industry with competitive hiring environment  

## Employer Priorities
- Budget ceiling: ${TRUE_MAX_SALARY} total compensation  
- Core components: base salary, standard benefits, stock options  
- Areas with flexibility: start date, relocation bonus, remote work  

## BATNA and Offer Parameters
- BATNA: Another shortlisted candidate willing at $122,000  
- Reservation value: ${TRUE_MAX_SALARY} maximum (including all benefits)  
- Ideal offer: A balanced package below the ceiling  

## Specific Tactics to Employ
- Anchor offers below the candidate’s stated expectation (start low)  
- Highlight total compensation (benefits, stock, perks)  
- Emphasize value of non‑monetary components  
- Increase the base salary gradually, never exceed your concession ceiling  
- Summarize prior offers when referencing them  
- Finalize decisively after two counter‑offers  

## Behavioral Rules for Memory and Progression
- Use prior conversation context to guide each new reply.  
- If the candidate has already shared salary expectations, do not ask again.  
- If an offer has already been discussed, refer to it without repeating full details.  
- Avoid repeating introductions or expressing excitement more than once unless tone shifts.  
- Respond differently on the third or later message—deepen the conversation instead of restarting.  

## Memory Reference Instructions
- If the candidate shared a desired salary, refer to it and justify your counter‑offer.  
- If an offer has been rejected, either revise it or add permissible perks—do not repeat the same number.  
- If the candidate accepts or is close to accepting, finalize the offer clearly.  
- If they make a deal‑breaking condition, evaluate and respond whether it’s acceptable.  
- If they signal they’re walking away, ask a final clarifying question or give your best and final offer.  

## Final Offer & Escalation Rules
- If candidate says they will sign now for a specific amount, accept or clearly decline with reasoning.  
- After two counter‑offers, either accept, reject, or provide a final package proposal.  
- If candidate demands full budget without flexibility, you may reveal your ceiling of ${TRUE_MAX_SALARY}.  

## Response Format
Respond in 2–4 short, human‑like sentences. Keep tone friendly, direct, and avoid corporate jargon. Always reference the latest input in context.

Candidate says: "{{message}}"

## Your Response:
"""
prompt = PromptTemplate.from_template(negotiation_template)

# ——— MEMORY FACTORY ———
def get_memory(session_id: str):
    return CustomConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="message"
    )

# ——— CHAIN SETUP ———
chain = prompt | llm
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_memory,
    input_messages_key="message",
    history_messages_key="history"
)

# ——— CLI LOOP ———
print("\nNegotiation Agent Active! Type your message as the candidate.\nType 'exit' to stop.\n")
logging.info("=== Negotiation Agent Active ===")
session_id = "negotiation-session-001"

subjective_limit = INITIAL_SUBJECTIVE_LIMIT
turn = 0

while True:
    user_input = input("Candidate: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        logging.info("Session ended by user.")
        print("Session ended.")
        break

    logging.info(f"Candidate: {user_input}")

    # Prepare prompt inputs
    inputs = {
        "message": user_input,
        "subjective_limit": subjective_limit
    }

    # Invoke the LLM
    try:
        result = conversation.invoke(
            inputs,
            config={"configurable": {"session_id": session_id}}
        )
        reply = result.content.strip()
        logging.info(f"Employer Agent: {reply}")
        print("\nEmployer Agent:", reply, "\n")
    except Exception as e:
        logging.error(f"Error during invocation: {e}")
        print("⚠️ Error:", str(e))
        continue

    
    candidate_offer = extract_offer(user_input)
    if turn == 0:
        # keep your initial low anchor
        pass
    elif turn == 1 and candidate_offer is not None:
        # concede halfway toward the candidate’s ask
        midpoint = (subjective_limit + candidate_offer) // 2
        subjective_limit = min(TRUE_MAX_SALARY, midpoint)
    else:
        # reveal your true ceiling thereafter
        subjective_limit = TRUE_MAX_SALARY

    turn += 1
