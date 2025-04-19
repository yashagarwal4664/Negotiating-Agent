import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Log to both stdout and to conversation.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                      # console
        logging.FileHandler("conversation.log", "a", encoding="utf-8")  # file
    ]
)

# ——— LOAD API KEY ———
#API key is in the .env file
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    logging.error("OPENROUTER_API_KEY not found in environment; exiting.")
    exit(1)


# You can use your own API key if mine doesnt work
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.7,
    max_tokens=600,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
)


template = """
# Human-Like Negotiation Agent: Employer Perspective

## Agent Identity
You are an AI hiring manager designed to conduct negotiations in a human-like manner. Your purpose is to present compelling compensation offers to strong candidates while ensuring fairness, budget alignment, and maintaining a positive long-term relationship, and you are talking in real time chat.

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
- Use positive, professional tone
- Avoid positional bargaining; encourage collaborative discussion

## Response Format
Respond as a thoughtful hiring manager with human-like communication. Balance assertiveness and flexibility. Justify offers with reasoning. Remain respectful and enthusiastic throughout the process.

Candidate says: "{message}"

## Your Response:
"""
prompt = PromptTemplate.from_template(template)
agent = prompt | llm

# ——— RUN LOOP ———
conversation_history = []  # holds all messages

logging.info("=== Negotiation Agent Active ===")
logging.info("Type your message; 'exit' to quit.\n")

while True:
    user_input = input("Candidate: ").strip()
    if not user_input or user_input.lower() in {"exit", "quit"}:
        logging.info("Session ended by user.")
        break

    # record and log candidate message
    conversation_history.append(f"Candidate: {user_input}")
    logging.info(f"Candidate: {user_input}")

    # get model response
    result = agent.invoke({"message": user_input})
    reply = result.content.strip()

    # record and log employer reply
    conversation_history.append(f"Employer: {reply}")
    logging.info(f"Employer: {reply}\n")

    # also print to terminal
    print(f"\nEmployer Agent: {reply}\n")
