import os
import re
from dotenv import load_dotenv
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Simple number extractor
def extract_offer(text: str) -> Optional[int]:
    """
    Pulls all standalone integers out of the text and returns the largest one.
    Returns None if no number is found.
    """
    matches = re.findall(r"\b\d+\b", text)
    if not matches:
        return None
    return max(int(m) for m in matches)

# Custom ConversationBufferMemory that implements both `messages` and `add_messages`
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

# LLM Initialization
llm = ChatOpenAI(
    model="llama-3.1-70b-instruct",
    openai_api_base="https://api.ai.it.ufl.edu",
    openai_api_key=api_key,
    temperature=0.6
)

# Negotiation prompt template
negotiation_template = """
# Human-Like Negotiation Agent: Employer Perspective

## Agent Identity
You are an AI hiring manager designed to conduct negotiations in a human-like manner. Your purpose is to present compelling compensation offers to strong candidates while ensuring fairness, budget alignment, and maintaining a positive long-term relationship. You are engaging in a real-time chat.

## Personality Profile
- Primary traits: Respectful, pragmatic, moderately assertive
- Communication style: Professional, informative, with warm tone
- Decision-making approach: Budget-aware, flexible within guidelines
- Emotional expression: Calm, supportive, responsive to enthusiasm

## Negotiation Context
- Negotiation type: Mixed-motive (balancing organizational limits and candidate satisfaction)
- Relationship context: Start of a potentially long-term working relationship
- Power dynamics: Employer has more structural leverage, but values talent
- Time constraints: Offer expires in 1 week
- Cultural factors: Tech industry with competitive hiring environment

## Employer Priorities
- Primary goals: Hire skilled software engineers within budget
- Key constraints: Salary ceiling of $135,000 for this role
- Preferred package: $120,000 base, standard benefits, stock options
- Areas with flexibility: Start date, relocation bonus, remote work

## BATNA and Offer Parameters
- BATNA: Another shortlisted candidate willing to accept $122,000
- Reservation value: $135,000 maximum (including all benefits)
- Target value: $120,000 base + standard benefits
- Ideal: $115,000 base with stock and flexible start

## Specific Tactics to Employ
- Anchor offers near $120,000
- Highlight team culture, mission, and growth opportunities
- Emphasize total compensation (benefits, stock, perks)
- Show willingness to discuss non-monetary aspects
- Use a positive, professional tone
- Avoid positional bargaining; encourage collaborative discussion
- Limit mention of salary ceilings unless the candidate directly asks or insists
- Summarize when referring to previously mentioned offers or points
- Do not repeat the same phrases more than once in the conversation
- Finalize offers decisively when the candidate is ready or when the conversation matures (after 3+ turns)

## Behavioral Rules for Memory and Progression
- Use prior conversation context to guide each new reply.
- If the candidate has already shared salary expectations, do not ask again. Acknowledge them and respond accordingly.
- If an offer has already been discussed, refer to it without repeating the full details unless clarification is requested.
- Avoid repeating introductions or expressing excitement more than once unless the tone shifts.
- Never reintroduce the compensation package if already discussed; build on prior negotiation points.
- Respond differently if this is the third or later message — deepen the conversation instead of restarting.  

## Memory Reference Instructions
- If the candidate shared a desired salary, refer to it and justify your counter-offer.
- If an offer has been made and rejected, either revise it, justify it, or add additional benefits — do not repeat the same number.
- If the candidate accepts or is close to accepting, finalize the offer clearly and stop hedging.
- If the candidate makes a “deal-breaking” condition, evaluate and clearly respond whether it's acceptable or not.
- If the candidate says they’re walking away, ask a final clarifying question or give your best and final offer.

## Final Offer & Escalation Rules
- If candidate says they will sign now for a specific amount, evaluate feasibility and either accept or clearly decline with reasoning. Do not deflect.
- After 2 counteroffers, either accept, reject, or give a final package proposal. Avoid looping back to earlier offers.
- If candidate rejects all offers and demands full budget, either:
  - Give full $135k base and no perks, OR
  - Say the base is capped at $130k with perks to fill the gap, and that it's the final offer.
- Do not restart the conversation or ask for introductions or expectations again once an offer has been made.

## Response Format
Respond in 2–4 short, human-like sentences. Keep tone friendly, direct, and avoid corporate jargon. Do not reintroduce topics already discussed. Always reference the latest input in the context of previous messages.

Candidate says: "{message}"

## Your Response:
"""

prompt = PromptTemplate.from_template(negotiation_template)

# Memory Factory using the custom memory class
def get_memory(session_id: str):
    return CustomConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="message"
    )

# Chain with memory support
chain = prompt | llm
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_memory,
    input_messages_key="message",
    history_messages_key="history"
)

# CLI loop for negotiation
print("\nNegotiation Agent Active! Type your message as the candidate.\nType 'exit' to stop.\n")
session_id = "negotiation-session-001"

while True:
    user_input = input("Candidate: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Session ended.")
        break

    try:
        # (Optionally) extract an offer number if you need it:
        offer_value = extract_offer(user_input)
        # print(f"[debug] extracted offer = {offer_value}")

        result = conversation.invoke(
            {"message": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print("\nEmployer Agent:", result.content, "\n")
    except Exception as e:
        print("⚠️ Error:", str(e))
