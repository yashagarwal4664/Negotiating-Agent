# Knowledge Graph for Negotiation State

import networkx as nx
import datetime
import json
from typing import Optional, Tuple, List, Dict, Any

class NegotiationKnowledgeGraph:
    def __init__(self, session_id: str, candidate_id: str = "candidate"):
        self.graph = nx.DiGraph()
        self.session_id = session_id
        self.candidate_id = candidate_id
        self.graph.add_node(session_id, type="NegotiationSession", start_time=datetime.datetime.now())
        self.graph.add_node(candidate_id, type="Candidate")
        self.graph.add_edge(session_id, candidate_id, type="PARTICIPANT")
        self.turn_count = 0

    def _get_turn_node_id(self, turn_number: int) -> str:
        return f"turn_{turn_number}"

    def _get_offer_node_id(self, turn_number: int, offered_by: str) -> str:
        return f"offer_{turn_number}_{offered_by}"

    def _get_limit_node_id(self, turn_number: int) -> str:
        return f"limit_{turn_number}"

    def _get_perk_node_id(self, perk_name: str) -> str:
        return f"perk_{perk_name.lower().replace(' ', '_')}"

    def add_turn(self, candidate_message: str, agent_response: str, current_subjective_limit: int) -> int:
        self.turn_count += 1
        turn_node_id = self._get_turn_node_id(self.turn_count)
        limit_node_id = self._get_limit_node_id(self.turn_count)
        
        self.graph.add_node(turn_node_id, 
                            type="Turn", 
                            turn_number=self.turn_count, 
                            candidate_message=candidate_message,
                            agent_response=agent_response,
                            timestamp=datetime.datetime.now())
        
        self.graph.add_node(limit_node_id,
                            type="Limit",
                            amount=current_subjective_limit,
                            turn_number=self.turn_count)

        self.graph.add_edge(self.session_id, turn_node_id, type="HAS_TURN")
        self.graph.add_edge(turn_node_id, limit_node_id, type="CURRENT_LIMIT")

        if self.turn_count > 1:
            prev_turn_node_id = self._get_turn_node_id(self.turn_count - 1)
            self.graph.add_edge(prev_turn_node_id, turn_node_id, type="PRECEDES")
            
        return self.turn_count

    def add_offer(self, turn_number: int, offer_details: Dict[str, Any], offered_by: str, status: str = "proposed") -> Optional[str]:
        turn_node_id = self._get_turn_node_id(turn_number)
        offer_node_id = self._get_offer_node_id(turn_number, offered_by)
        
        if not self.graph.has_node(turn_node_id):
            print(f"Warning: Turn node {turn_node_id} not found for adding offer.")
            return None
            
        self.graph.add_node(offer_node_id,
                            type="Offer",
                            details=offer_details, 
                            offered_by=offered_by,
                            status=status,
                            turn_number=turn_number)
        
        self.graph.add_edge(turn_node_id, offer_node_id, type="CONTAINS_OFFER")
        
        if offered_by == "agent":
            agent_response_node = f"agent_response_{turn_number}"
            if not self.graph.has_node(agent_response_node):
                 self.graph.add_node(agent_response_node, type="AgentResponse", turn=turn_number)
                 self.graph.add_edge(turn_node_id, agent_response_node, type="HAS_RESPONSE")
            self.graph.add_edge(agent_response_node, offer_node_id, type="JUSTIFIES")
            
        return offer_node_id

    def update_offer_status(self, offer_node_id: str, status: str):
        if self.graph.has_node(offer_node_id) and self.graph.nodes[offer_node_id].get("type") == "Offer":
            self.graph.nodes[offer_node_id]["status"] = status
            if status == "rejected":
                 self.graph.add_edge(self.candidate_id, offer_node_id, type="REJECTED")
            elif status == "accepted":
                 self.graph.add_edge(self.candidate_id, offer_node_id, type="ACCEPTED")
        else:
            print(f"Warning: Offer node {offer_node_id} not found for status update.")

    def add_candidate_preference(self, perk_name: str):
        perk_node_id = self._get_perk_node_id(perk_name)
        if not self.graph.has_node(perk_node_id):
            self.graph.add_node(perk_node_id, type="Perk", name=perk_name)
        # Avoid adding duplicate preference edges
        if not self.graph.has_edge(self.candidate_id, perk_node_id):
            self.graph.add_edge(self.candidate_id, perk_node_id, type="PREFERS")

    def get_candidate_preferences(self) -> List[str]:
        preferences = []
        if self.graph.has_node(self.candidate_id):
            for _, target, data in self.graph.out_edges(self.candidate_id, data=True):
                if data.get("type") == "PREFERS":
                    if self.graph.has_node(target) and self.graph.nodes[target].get("type") == "Perk":
                        preferences.append(self.graph.nodes[target].get("name", "Unknown Perk"))
        return preferences

    def get_last_offer_details(self, offered_by: Optional[str] = None) -> Optional[Tuple[int, Dict[str, Any], str]]:
        offers = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Offer":
                if offered_by is None or data.get("offered_by") == offered_by:
                    details = data.get("details")
                    if isinstance(details, dict):
                         offers.append((data.get("turn_number", 0), details, node))
        
        if not offers:
            return None
        
        offers.sort(key=lambda x: x[0], reverse=True)
        return offers[0]
        
    def get_offers_by_status(self, status: str, offered_by: Optional[str] = None) -> List[Tuple[int, Dict[str, Any], str]]:
        matching_offers = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "Offer" and data.get("status") == status:
                if offered_by is None or data.get("offered_by") == offered_by:
                    details = data.get("details")
                    if isinstance(details, dict):
                        matching_offers.append((data.get("turn_number", 0), details, node))
        matching_offers.sort(key=lambda x: x[0], reverse=True)
        return matching_offers

    def get_current_limit(self) -> Optional[int]:
         if self.turn_count == 0:
             return None
         limit_node_id = self._get_limit_node_id(self.turn_count)
         if self.graph.has_node(limit_node_id):
             limit_amount = self.graph.nodes[limit_node_id].get("amount")
             if isinstance(limit_amount, int):
                 return limit_amount
         if self.turn_count > 1:
             prev_limit_node_id = self._get_limit_node_id(self.turn_count - 1)
             if self.graph.has_node(prev_limit_node_id):
                 prev_limit_amount = self.graph.nodes[prev_limit_node_id].get("amount")
                 if isinstance(prev_limit_amount, int):
                     return prev_limit_amount
         return None

    def get_negotiation_summary(self) -> str:
        summary = f"Negotiation Summary (Session: {self.session_id}, Turns: {self.turn_count}):\n"
        prefs = self.get_candidate_preferences()
        if prefs:
            summary += f"Candidate Preferences: {', '.join(prefs)}\n"
            
        turns = sorted([ (data["turn_number"], node_id) 
                         for node_id, data in self.graph.nodes(data=True) 
                         if data.get("type") == "Turn"], key=lambda x: x[0])

        for turn_num, turn_node_id in turns:
            turn_data = self.graph.nodes[turn_node_id]
            summary += f"\nTurn {turn_num}:\n"
            summary += f"  Candidate: {turn_data.get('candidate_message', 'N/A')}\n"
            summary += f"  Agent: {turn_data.get('agent_response', 'N/A')}\n"
            
            limit = "N/A"
            limit_node_id = self._get_limit_node_id(turn_num)
            if self.graph.has_node(limit_node_id):
                limit = self.graph.nodes[limit_node_id].get("amount", "N/A")
            summary += f"  Agent Limit for this turn: ${limit}\n"

            offers_in_turn = []
            for neighbor in self.graph.successors(turn_node_id):
                neighbor_data = self.graph.nodes[neighbor]
                edge_data = self.graph.get_edge_data(turn_node_id, neighbor)
                if edge_data.get("type") == "CONTAINS_OFFER":
                    offer_details = neighbor_data.get("details", {})
                    offered_by = neighbor_data.get("offered_by", "unknown")
                    status = neighbor_data.get("status", "proposed")
                    offers_in_turn.append(f"  {offered_by.capitalize()} Offer ({status}): {json.dumps(offer_details)}")
            
            if offers_in_turn:
                summary += "\n".join(offers_in_turn) + "\n"
                 
        return summary

    def add_similar_offer_relation(self, offer_node_id_1: str, offer_node_id_2: str):
        if self.graph.has_node(offer_node_id_1) and self.graph.has_node(offer_node_id_2):
            if self.graph.nodes[offer_node_id_1].get("type") == "Offer" and self.graph.nodes[offer_node_id_2].get("type") == "Offer":
                 # Avoid self-loops and duplicate edges
                 if offer_node_id_1 != offer_node_id_2 and not self.graph.has_edge(offer_node_id_1, offer_node_id_2, key="SIMILAR_TO") and not self.graph.has_edge(offer_node_id_2, offer_node_id_1, key="SIMILAR_TO"):
                     self.graph.add_edge(offer_node_id_1, offer_node_id_2, type="SIMILAR_TO")
            else:
                 print(f"Warning: One or both nodes ({offer_node_id_1}, {offer_node_id_2}) are not Offer nodes.")
        else:
            print(f"Warning: One or both nodes ({offer_node_id_1}, {offer_node_id_2}) not found for adding similarity relation.")

    def get_similar_offers(self, offer_node_id: str) -> List[str]:
        similar = []
        if self.graph.has_node(offer_node_id) and self.graph.nodes[offer_node_id].get("type") == "Offer":
            # Check both incoming and outgoing similarity edges
            for u, v, data in self.graph.edges(data=True):
                if data.get("type") == "SIMILAR_TO":
                    if u == offer_node_id:
                        similar.append(v)
                    elif v == offer_node_id:
                        similar.append(u)
        return list(set(similar)) # Return unique list

# Example Usage (for testing)
if __name__ == "__main__":
    kg = NegotiationKnowledgeGraph("test_session_kg_123", candidate_id="test_candidate")
    
    limit1 = 115000
    t1 = kg.add_turn("I need at least $130k and remote work.", "We can start with $115k base, standard benefits.", limit1)
    offer1_cand_details = {"base": 130000, "perks": ["remote work"]}
    offer1_agent_details = {"base": 115000, "benefits": "standard"}
    o1c = kg.add_offer(t1, offer1_cand_details, "candidate")
    o1a = kg.add_offer(t1, offer1_agent_details, "agent")
    kg.add_candidate_preference("remote work")
    print(f"Current Limit after Turn 1: {kg.get_current_limit()}")
    print(f"Last Candidate Offer: {kg.get_last_offer_details('candidate')}")
    print(f"Candidate Preferences: {kg.get_candidate_preferences()}")

    limit2 = 122500
    t2 = kg.add_turn("115k is too low. How about $125k with remote?", "Okay, how about $120k base and we can approve remote work?", limit2)
    offer2_cand_details = {"base": 125000, "perks": ["remote work"]}
    offer2_agent_details = {"base": 120000, "perks": ["remote work"], "benefits": "standard"}
    o2c = kg.add_offer(t2, offer2_cand_details, "candidate")
    o2a = kg.add_offer(t2, offer2_agent_details, "agent")
    kg.update_offer_status(o1a, "rejected") # Candidate implicitly rejected agent's first offer
    print(f"Current Limit after Turn 2: {kg.get_current_limit()}")
    print(f"Rejected Agent Offers: {kg.get_offers_by_status('rejected', 'agent')}")

    limit3 = 123000 
    t3 = kg.add_turn("Okay, $120k + remote is better. Can we finalize at $123k?", "Yes, we can agree to $123k base salary with remote work. Deal?", limit3)
    offer3_cand_details = {"base": 123000, "perks": ["remote work"]}
    offer3_agent_details = {"base": 123000, "perks": ["remote work"], "benefits": "standard"}
    o3c = kg.add_offer(t3, offer3_cand_details, "candidate")
    o3a = kg.add_offer(t3, offer3_agent_details, "agent", status="accepted")
    kg.update_offer_status(o2a, "superseded") # Agent's second offer is no longer active
    kg.update_offer_status(o3a, "accepted") # Explicitly mark final offer as accepted
    print(f"Current Limit after Turn 3: {kg.get_current_limit()}")
    print(f"Accepted Agent Offers: {kg.get_offers_by_status('accepted', 'agent')}")

    kg.add_similar_offer_relation(o1a, o2a) # Example similarity
    print(f"Offers similar to {o1a}: {kg.get_similar_offers(o1a)}")

    print("\n--- Negotiation Summary ---")
    print(kg.get_negotiation_summary())

