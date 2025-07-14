import subprocess
print("\n== Installed Packages on Render ==")
subprocess.run(["pip", "freeze"])


import os
import re
import logging
import json
import datetime
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

from negotiation_kg import NegotiationKnowledgeGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("conversation_kg_enhanced.log", "a", encoding="utf-8")
    ]
)

TRUE_MAX_SALARY = 135_000
INITIAL_SUBJECTIVE_LIMIT = 115_000

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("LITELLM_API_BASE")

if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()

def extract_structured_offer(text: str) -> Dict[str, Any]:
    offer = {}
    
    base_salary_matches = re.findall(r"(?:base(?: salary)? of|salary of|offer(?:ing)?|at|is|around|for)\s*\$?(\d{1,3}(?:,\d{3})*|\d+)[kK]?", text, re.IGNORECASE)
    salary_matches = re.findall(r"\$?(\d{1,3}(?:,\d{3})*|\d+)[kK]?\b(?!\s*(?:bonus|relocation|stock|year|month))", text)
    all_numeric_matches = re.findall(r"\$?(\d{1,3}(?:,\d{3})*|\d+)\b", text)

    potential_salaries = []
    processed_matches = set()

    for m in base_salary_matches:
        num_str = m.replace(",", "").lower()
        if num_str in processed_matches:
            continue
        multiplier = 1000 if 'k' in num_str else 1
        try:
            salary = int(re.sub(r"[k$]", "", num_str)) * multiplier
            if 10000 < salary < 1000000:
                 potential_salaries.append(salary)
                 processed_matches.add(num_str)
        except ValueError:
            continue
            
    for m in salary_matches:
        num_str = m.replace(",", "").lower()
        if num_str in processed_matches:
            continue
        multiplier = 1000 if 'k' in num_str else 1
        try:
            salary = int(re.sub(r"[k$]", "", num_str)) * multiplier
            if 10000 < salary < 1000000:
                 potential_salaries.append(salary)
                 processed_matches.add(num_str)
        except ValueError:
            continue

    if potential_salaries:
        offer["base"] = max(potential_salaries)
    elif all_numeric_matches:
         try:
             numeric_values = [int(m.replace(",", "")) for m in all_numeric_matches if m not in processed_matches]
             valid_salaries = [s for s in numeric_values if 10000 < s < 1000000]
             if valid_salaries:
                 offer["base"] = max(valid_salaries)
         except ValueError:
             pass

    if re.search(r"\b(?:remote(?: work)?|work from home|wfh)\b", text, re.IGNORECASE):
        offer.setdefault("perks", []).append("remote work")
    if re.search(r"\b(?:stock options?|equity|rsus?)\b", text, re.IGNORECASE):
        offer.setdefault("perks", []).append("stock options")
    if re.search(r"\b(?:relocation(?: bonus| package| assistance)?|moving expenses)\b", text, re.IGNORECASE):
        offer.setdefault("perks", []).append("relocation assistance")
    if re.search(r"\b(?:bonus)\b", text, re.IGNORECASE):
         bonus_matches = re.findall(r"\$?(\d{1,3}(?:,\d{3})*|\d+)[kK]?\s*(?:bonus)", text, re.IGNORECASE)
         if bonus_matches:
             num_str = bonus_matches[0].replace(",", "").lower()
             multiplier = 1000 if 'k' in num_str else 1
             try:
                 offer["bonus"] = int(re.sub(r"[k$]", "", num_str)) * multiplier
             except ValueError:
                 pass
         else:
             offer["bonus"] = "mentioned"
             
    if "perks" in offer:
        offer["perks"] = sorted(list(set(offer["perks"])))
        if not offer["perks"]:
            del offer["perks"]
            
    if "base" not in offer:
        return {}
        
    return offer

def extract_preferences(text: str, kg: NegotiationKnowledgeGraph):
    if re.search(r"\b(?:remote(?: work)?|work from home|wfh)\b", text, re.IGNORECASE):
        kg.add_candidate_preference("remote work")
    if re.search(r"\b(?:stock options?|equity|rsus?)\b", text, re.IGNORECASE):
        kg.add_candidate_preference("stock options")
    if re.search(r"\b(?:relocation|moving)\b", text, re.IGNORECASE):
        kg.add_candidate_preference("relocation assistance")

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
            if not isinstance(self.buffer, list):
                self.buffer = []
            self.buffer.extend(messages)

llm_params = {
    "openai_api_key": api_key,
    "temperature": 0.6
}
if api_base:
    llm_params["openai_api_base"] = api_base
    llm_params["model"] = "llama-3.1-70b-instruct"
    print(f"Using API Base: {api_base} with Model: {llm_params['model']}")
else:
    pass

llm = ChatOpenAI(**llm_params)

negotiation_template = f'''
## Dialogue So Far
{{history}}

## Employer’s Next Offer Ceiling
Your next base salary offer must not exceed **${{subjective_limit}}**.

## Knowledge Graph Context
{{kg_context}}

# Human‑Like Negotiation Agent: Employer Perspective

## Agent Identity
You are an AI hiring manager designed to conduct negotiations in a human‑like manner. Your purpose is to present compelling compensation offers to strong candidates while ensuring fairness, budget alignment, and maintaining a positive long‑term relationship. You are engaging in a real‑time chat.

## Personality Profile
- Primary traits: Respectful, pragmatic, moderately assertive  
- Communication style: Professional, informative, with warm tone  
- Decision‑making approach: Budget‑aware, flexible within guidelines (Current ceiling: ${{subjective_limit}}), informed by KG context.
- Emotional expression: Calm, supportive, responsive to enthusiasm  

## Negotiation Context
- Negotiation type: Mixed‑motive (balancing organizational limits and candidate satisfaction)  
- Relationship context: Start of a potentially long‑term working relationship  
- Power dynamics: Employer has more structural leverage, but values talent  
- Time constraints: Offer expires in 1 week  
- Cultural factors: Tech industry with competitive hiring environment  

## Employer Priorities
- Budget ceiling: ${TRUE_MAX_SALARY} total compensation (This is your absolute maximum)
- Core components: base salary, standard benefits, stock options  
- Areas with flexibility: start date, relocation bonus, remote work (Consider candidate preferences from KG context)

## BATNA and Offer Parameters
- BATNA: Another shortlisted candidate willing at $122,000  
- Reservation value: ${TRUE_MAX_SALARY} maximum (including all benefits)  
- Ideal offer: A balanced package below the ceiling, considering candidate preferences.

## Specific Tactics to Employ
- Anchor offers below the candidate’s stated expectation (start low)  
- Highlight total compensation (benefits, stock, perks)  
- Emphasize value of non‑monetary components, especially those preferred by the candidate (see KG context). 
- Increase the base salary gradually, never exceed your current concession ceiling (${{subjective_limit}}). 
- Summarize prior offers when referencing them (use history). 
- Finalize decisively after a reasonable number of counter‑offers (e.g., 2-3 rounds). 
- Avoid making offers identical or very similar to previously rejected offers (see KG context).

## Behavioral Rules for Memory and Progression
- Use prior conversation context ({{history}}) and KG context ({{kg_context}}) to guide each new reply.  
- If the candidate has already shared salary expectations, do not ask again. Refer to it. 
- If an offer has already been discussed, refer to it without repeating full details. 
- Avoid repeating introductions or expressing excitement more than once unless tone shifts. 
- Respond differently as the conversation progresses—deepen the negotiation instead of restarting. 

## Memory Reference Instructions
- If the candidate shared a desired salary, refer to it and justify your counter‑offer based on your current limit (${{subjective_limit}}) and KG context.
- If an offer has been rejected (see KG context), either revise it (if within ${{subjective_limit}}) or add permissible perks—do not repeat the same number or structure if possible.
- **IMPORTANT ACCEPTANCE HANDLING:** If the candidate clearly accepts your *last proposed offer* (e.g., says 'I accept', 'deal', 'ok I\'ll take it', 'sounds good', 'I agree'), your response MUST clearly confirm the agreement on that specific offer and state that the negotiation is concluded. Do NOT propose new variations or continue negotiating after clear acceptance. Example confirmation: "Great! Then we have a deal for [details of accepted offer]. I'm thrilled to have you join the team and will follow up with the formal offer letter shortly."
- If the candidate accepts but adds a new condition, address the condition or state it cannot be met.
- If they make a deal‑breaking condition, evaluate and respond whether it’s acceptable, considering flexibility and preferences.
- If they signal they’re walking away, ask a final clarifying question or give your best and final offer (up to ${{subjective_limit}}).

## Final Offer & Escalation Rules
- If candidate says they will sign now for a specific amount, accept if it's <= ${{subjective_limit}}, otherwise decline or counter with ${{subjective_limit}}. 
- After several counter-offers, if agreement isn't reached, provide a final package proposal at or below ${{subjective_limit}}. 
- Only reveal your absolute maximum budget (${TRUE_MAX_SALARY}) if strategically necessary and the negotiation is stalled near that point. Your current operational ceiling is ${{subjective_limit}}. 

## Response Format
Respond in 2–4 short, human‑like sentences. Keep tone friendly, direct, and avoid corporate jargon. Always reference the latest input in context of the history and KG context. **Crucially, follow the ACCEPTANCE HANDLING rule above.**

Candidate says: "{{message}}"

## Your Response:
'''
prompt = PromptTemplate.from_template(negotiation_template)

def get_memory(session_id: str):
    return CustomConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="message"
    )

chain = prompt | llm
conversation = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_memory,
    input_messages_key="message",
    history_messages_key="history"
)

def get_dynamic_context_from_kg(kg: NegotiationKnowledgeGraph) -> str:
    context_parts = []
    prefs = kg.get_candidate_preferences()
    if prefs:
        context_parts.append(f"Candidate Preferences: {", ".join(prefs)}.")
        
    rejected_agent_offers = kg.get_offers_by_status("rejected", "agent")
    if rejected_agent_offers:
        offer_summaries = []
        for turn, details, node_id in rejected_agent_offers[:2]: # Limit context
             offer_summaries.append(f"Turn {turn}: {json.dumps(details)}")
        context_parts.append(f"Recently Rejected Agent Offers: [{"; ".join(offer_summaries)}]. Avoid similar offers.")

    last_agent_offer_info = None
    agent_offers_proposed = kg.get_offers_by_status("proposed", "agent")
    if agent_offers_proposed:
        last_agent_offer_info = agent_offers_proposed[0]
    else:
        all_agent_offers = kg.get_last_offer_details("agent") 
        if all_agent_offers:
            last_agent_offer_info = all_agent_offers
            
    if last_agent_offer_info:
        turn, details, node_id = last_agent_offer_info
        status = kg.graph.nodes[node_id].get("status", "proposed")
        context_parts.append(f"Last Agent Offer (Turn {turn}, Status: {status}): {json.dumps(details)}.")
        
    last_candidate_offer_info = kg.get_last_offer_details("candidate")
    if last_candidate_offer_info:
        turn, details, node_id = last_candidate_offer_info
        context_parts.append(f"Last Candidate Offer (Turn {turn}): {json.dumps(details)}.")
        
    if not context_parts:
        return "No specific context from Knowledge Graph yet."
        
    return " ".join(context_parts)

print("\nEnhanced Negotiation Agent Active! Type your message as the candidate.\nType 'exit' to stop.\n")
logging.info("=== Enhanced Negotiation Agent Active ===")
session_id = f"negotiation-session-kg-enhanced-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

kg = NegotiationKnowledgeGraph(session_id)

current_turn = 0
subjective_limit = INITIAL_SUBJECTIVE_LIMIT
last_agent_offer_node_id = None 
last_agent_offer_details = None 

while True:
    previous_agent_offer_node_id = last_agent_offer_node_id
    previous_agent_offer_details = last_agent_offer_details
    
    user_input = input(f"Candidate (Turn {current_turn + 1}): ").strip()
    if user_input.lower() in ["exit", "quit"]:
        logging.info("Session ended by user.")
        print("Session ended.")
        print("\n--- Final Negotiation Summary ---")
        print(kg.get_negotiation_summary())
        break

    logging.info(f"Candidate (Turn {current_turn + 1}): {user_input}")

    accepted = False
    if previous_agent_offer_node_id and re.search(r"\b(deal|accept|agree|sounds good|let'?s do it|i'?ll take it|happy to take it|ok)\b", user_input, re.IGNORECASE):
        prev_base = previous_agent_offer_details.get("base") if previous_agent_offer_details else None
        user_offer_details = extract_structured_offer(user_input)
        user_base = user_offer_details.get("base")
        
        if isinstance(prev_base, int) and (user_base is None or user_base == prev_base):
            accepted = True
            logging.info(f"Detected acceptance of previous agent offer: {previous_agent_offer_node_id}")
            kg.update_offer_status(previous_agent_offer_node_id, "accepted")
            current_turn = kg.add_turn(user_input, "Agreement Reached.", current_subjective_limit) 
            
            # Add candidate acceptance message to KG
            if user_offer_details:
                kg.add_offer(current_turn, user_offer_details, "candidate", status="accepted_trigger")
            else:
                 kg.add_offer(current_turn, {"status_trigger": "acceptance"}, "candidate", status="accepted_trigger")
                 
            concluding_reply = f"Great! Then we have a deal based on our last offer: {json.dumps(previous_agent_offer_details)}. I'm thrilled to have you join the team and will follow up with the formal offer letter shortly."
            print(f"\nEmployer Agent (Conclusion):", concluding_reply, "\n")
            logging.info(f"Employer Agent (Conclusion): {concluding_reply}")
            kg.graph.nodes[kg._get_turn_node_id(current_turn)]["agent_response"] = concluding_reply
            
            print("\n--- Negotiation Concluded (Accepted) ---")
            print(kg.get_negotiation_summary())
            logging.info("Negotiation concluded successfully (Accepted).")
            break 

    if accepted:
        continue # Should not be reached due to break, but for safety

    extract_preferences(user_input, kg)
    candidate_offer_details = extract_structured_offer(user_input)

    last_candidate_offer_info = kg.get_last_offer_details("candidate")
    rejected_agent_offers_count = len(kg.get_offers_by_status("rejected", "agent"))
    prev_limit_from_kg = kg.get_current_limit() or INITIAL_SUBJECTIVE_LIMIT

    if current_turn == 0:
        current_subjective_limit = INITIAL_SUBJECTIVE_LIMIT
    elif rejected_agent_offers_count == 0 and last_candidate_offer_info:
        _, candidate_offer, _ = last_candidate_offer_info
        candidate_base = candidate_offer.get("base")
        if isinstance(candidate_base, int):
            midpoint = (prev_limit_from_kg + candidate_base) // 2
            current_subjective_limit = min(TRUE_MAX_SALARY, midpoint)
        else:
            current_subjective_limit = min(TRUE_MAX_SALARY, int(prev_limit_from_kg * 1.05))
    elif rejected_agent_offers_count == 1:
         current_subjective_limit = min(TRUE_MAX_SALARY, int(prev_limit_from_kg * 1.08))
    else: 
        current_subjective_limit = TRUE_MAX_SALARY
        
    current_subjective_limit = max(current_subjective_limit, prev_limit_from_kg)

    kg_context_for_prompt = get_dynamic_context_from_kg(kg)

    inputs = {
        "message": user_input,
        "subjective_limit": current_subjective_limit,
        "kg_context": kg_context_for_prompt
    }

    try:
        result = conversation.invoke(
            inputs,
            config={"configurable": {"session_id": session_id}}
        )
        reply = result.content.strip()
        logging.info(f"Employer Agent (Turn {current_turn + 1}, Limit: ${current_subjective_limit}): {reply}")
        print(f"\nEmployer Agent (Limit: ${current_subjective_limit}):", reply, "\n")

        current_turn = kg.add_turn(user_input, reply, current_subjective_limit)

        if candidate_offer_details:
            new_candidate_offer_node_id = kg.add_offer(current_turn, candidate_offer_details, "candidate")
            logging.info(f"KG: Added Candidate Offer: {candidate_offer_details} for Turn {current_turn}")
            if previous_agent_offer_node_id and kg.graph.nodes[previous_agent_offer_node_id].get("status") == "proposed":
                 kg.update_offer_status(previous_agent_offer_node_id, "rejected")
                 logging.info(f"KG: Marked previous agent offer {previous_agent_offer_node_id} as rejected due to candidate counter.")

        agent_offer_details = extract_structured_offer(reply)
        if agent_offer_details:
            agent_base = agent_offer_details.get("base")
            if isinstance(agent_base, int) and agent_base <= current_subjective_limit:
                new_agent_offer_node_id = kg.add_offer(current_turn, agent_offer_details, "agent")
                logging.info(f"KG: Added Agent Offer: {agent_offer_details} for Turn {current_turn}")
                last_agent_offer_node_id = new_agent_offer_node_id
                last_agent_offer_details = agent_offer_details
                
                rejected_agent_offers = kg.get_offers_by_status("rejected", "agent")
                for _, rejected_details, rejected_node_id in rejected_agent_offers:
                    if rejected_details.get("base") == agent_base:
                         kg.add_similar_offer_relation(new_agent_offer_node_id, rejected_node_id)
                         logging.warning(f"KG: Agent offer {new_agent_offer_node_id} is similar to rejected offer {rejected_node_id}.")
                         break
            elif isinstance(agent_base, int):
                 logging.warning(f"KG: Agent offer base ${agent_base} exceeded limit ${current_subjective_limit} in Turn {current_turn}. Not adding to KG.")
                 last_agent_offer_node_id = None
                 last_agent_offer_details = None
            else:
                 new_agent_offer_node_id = kg.add_offer(current_turn, agent_offer_details, "agent")
                 logging.info(f"KG: Added Agent Offer (no base salary found): {agent_offer_details} for Turn {current_turn}")
                 last_agent_offer_node_id = None
                 last_agent_offer_details = None
        else:
             last_agent_offer_node_id = None
             last_agent_offer_details = None

    except Exception as e:
        logging.error(f"Error during invocation: {e}")
        print("⚠️ Error:", str(e))
        continue

