"""
Knowledge Graph Example: Demonstrating hallucination mitigation using knowledge graphs

This simplified example shows how to create and use a knowledge graph to verify
and enhance LLM outputs without requiring external API calls. You can use this as a
rough template if you want to implement this with LLM calls. 

To run:
    uv run python src/ember/examples/data/knowledge_graph_example.py
"""

from typing import Dict, List, Set
import random
from dataclasses import dataclass
from collections import defaultdict

from ember.api.operators import Operator, Specification, EmberModel

@dataclass
class Entity:
    """Represents a node in the knowledge graph."""
    id: str
    name: str
    type: str
    attributes: Dict[str, str]

@dataclass
class Relationship:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float

class KnowledgeGraph:
    """Simple in-memory knowledge graph implementation."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self._relationship_index = defaultdict(list)
        
    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity
        
    def add_relationship(self, relationship: Relationship) -> None:
        self.relationships.append(relationship)
        self._relationship_index[relationship.source_id].append(relationship)
        
    def get_related_entities(self, entity_id: str, max_depth: int = 2) -> Set[str]:
        """Get all entities connected to the given entity up to max_depth."""
        related = set()
        to_visit = [(entity_id, 0)]
        visited = set()
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            if current_id in visited or depth > max_depth:
                continue
                
            visited.add(current_id)
            related.add(current_id)
            
            # Add connected entities
            for rel in self._relationship_index[current_id]:
                to_visit.append((rel.target_id, depth + 1))
                
        return related
        
    def verify_statement(self, subject: str, predicate: str, object: str) -> float:
        """Verify a statement against the knowledge graph."""
        for rel in self.relationships:
            if (rel.source_id == subject and 
                rel.relation_type == predicate and 
                rel.target_id == object):
                return rel.confidence
        return 0.0

class EntityExtractionInput(EmberModel):
    """Input for entity extraction."""
    text: str

class EntityExtractionOutput(EmberModel):
    """Output from entity extraction."""
    entities: List[Entity]

class EntityExtractionSpec(Specification):
    """Specification for entity extraction."""
    input_model: type[EntityExtractionInput] = EntityExtractionInput
    structured_output: type[EntityExtractionOutput] = EntityExtractionOutput
    # In a real implementation, this could include a prompt template
    # prompt_template = "Extract entities from the following text:\n\n{text}"

class EntityExtractorOperator(Operator[EntityExtractionInput, EntityExtractionOutput]):
    """Extracts entities from text (simulated in this example)."""
    
    specification = EntityExtractionSpec()
    
    def forward(self, *, inputs: EntityExtractionInput) -> EntityExtractionOutput:
        """Extract entities from text."""
        keywords = {
            "Einstein": ("scientist", {"field": "physics"}),
            "physics": ("field", {"domain": "science"}),
            "photosynthesis": ("process", {"domain": "biology"}),
            "plants": ("organism", {"kingdom": "plantae"}),
            "DNA": ("molecule", {"type": "nucleic_acid"}),
        }
        
        entities = []
        for keyword, (type_, attrs) in keywords.items():
            if keyword.lower() in inputs.text.lower():
                entities.append(Entity(
                    id=f"{keyword.lower()}_1",
                    name=keyword,
                    type=type_,
                    attributes=attrs
                ))
        return EntityExtractionOutput(entities=entities)

class RelationMapperOperator:
    """Maps relationships between entities (simulated in this example)."""
    
    def __call__(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Map relationships between entities."""
        relationships = []
        
        # Predefined relationships for demonstration
        relationship_patterns = {
            ("einstein_1", "physics_1"): "contributed_to",
            ("photosynthesis_1", "plants_1"): "occurs_in",
        }
        
        # Create relationships between entities that appear together
        entity_ids = [e.id for e in entities]
        for (source, target), relation in relationship_patterns.items():
            if source in entity_ids and target in entity_ids:
                relationships.append(Relationship(
                    source_id=source,
                    target_id=target,
                    relation_type=relation,
                    confidence=0.9
                ))
                
        return relationships

class FactVerifierOperator:
    """Verifies generated text against knowledge graph."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        
    def __call__(self, inputs: Dict[str, str]) -> Dict[str, float]:
        """Verify facts in generated text."""
        generated_text = inputs.get("response", "")
        
        explicit_statements = self._extract_statements(generated_text)
        implicit_statements = self._extract_implicit_claims(generated_text)
        
        results = {}
        
        # Verify explicit statements against the knowledge graph
        for subject, predicate, object_ in explicit_statements:
            confidence = self.knowledge_graph.verify_statement(subject, predicate, object_)
            if subject in self.knowledge_graph.entities and object_ in self.knowledge_graph.entities:
                statement = f"{self.knowledge_graph.entities[subject].name} {predicate} {self.knowledge_graph.entities[object_].name}"
                results[statement] = confidence
        
        # Check implicit claims against the knowledge graph
        for claim, entities in implicit_statements:
            confidence = self._verify_implicit_claim(claim, entities)
            results[claim] = confidence
            
        return results
    
    def _extract_statements(self, text: str) -> List[tuple]:
        """Extract explicit subject-predicate-object statements from text."""
        statements = []
        
        if "einstein" in text.lower() and "physics" in text.lower():
            statements.append(("einstein_1", "contributed_to", "physics_1"))
        if "relativity" in text.lower() and "einstein" in text.lower():
            statements.append(("einstein_1", "discovered", "relativity_1"))
        if "photosynthesis" in text.lower() and "plants" in text.lower():
            statements.append(("photosynthesis_1", "occurs_in", "plants_1"))
            
        return statements
    
    def _extract_implicit_claims(self, text: str) -> List[tuple]:
        """Extract implicit claims that need verification."""
        claims = []
        
        if "einstein" in text.lower() and "telephone" in text.lower():
            claims.append(("Einstein invented the telephone", ["einstein_1"]))
        
        if "einstein" in text.lower() and "discovered" in text.lower() and "photosynthesis" in text.lower():
            claims.append(("Einstein discovered photosynthesis", ["einstein_1", "photosynthesis_1"]))
            
        return claims
    
    def _verify_implicit_claim(self, claim: str, entity_ids: List[str]) -> float:
        """Verify an implicit claim using the knowledge graph structure."""
        
        if "telephone" in claim.lower() and "einstein_1" in entity_ids:
            # Check if Einstein has any "invented" relationships
            for rel in self.knowledge_graph.relationships:
                if rel.source_id == "einstein_1" and rel.relation_type == "invented":
                    return 0.3

            return 0.1
            
        # For the Einstein/photosynthesis example:
        if "photosynthesis" in claim.lower() and "einstein_1" in entity_ids and "photosynthesis_1" in entity_ids:
            # Check if Einstein has any relationship with photosynthesis
            for rel in self.knowledge_graph.relationships:
                if (rel.source_id == "einstein_1" and rel.target_id == "photosynthesis_1") or \
                   (rel.target_id == "einstein_1" and rel.source_id == "photosynthesis_1"):
                    return rel.confidence
            # No relationship found between Einstein and photosynthesis
            return 0.0
            
        # Default case - we don't have enough information
        return 0.5

def create_mock_knowledge_graph() -> KnowledgeGraph:
    """Create a mock knowledge graph for demonstration."""
    kg = KnowledgeGraph()
    
    entities = [
        Entity("einstein_1", "Einstein", "scientist", {"field": "physics"}),
        Entity("physics_1", "Physics", "field", {"domain": "science"}),
        Entity("relativity_1", "Relativity", "theory", {"domain": "physics"}),
        Entity("photosynthesis_1", "Photosynthesis", "process", {"domain": "biology"}),
        Entity("plants_1", "Plants", "organism", {"kingdom": "plantae"}),
        Entity("dna_1", "DNA", "molecule", {"type": "nucleic_acid"}),
    ]
    
    for entity in entities:
        kg.add_entity(entity)
    
    relationships = [
        Relationship("einstein_1", "physics_1", "contributed_to", 1.0),
        Relationship("einstein_1", "relativity_1", "discovered", 1.0),
        Relationship("photosynthesis_1", "plants_1", "occurs_in", 1.0),
    ]
    
    for rel in relationships:
        kg.add_relationship(rel)
    
    return kg

def simulate_llm_response(inputs: Dict) -> Dict[str, str]:
    """Simulate an LLM response (with potential hallucinations)."""
    prompt = inputs.get("text", "")
    
    responses = {
        "Einstein": (
            "Einstein made significant contributions to physics, "
            "particularly in the development of the theory of relativity. "
            "His work revolutionized our understanding of space and time."
        ),
        "photosynthesis": (
            "Photosynthesis is a crucial process that occurs in plants, "
            "converting sunlight into energy. This fundamental process "
            "enables plants to produce their own food."
        ),
        "DNA": (
            "DNA is a double-helix molecule that contains genetic information. "
            "It serves as the blueprint for life, though our knowledge of its "
            "structure came long after Einstein's discoveries."  # Adding a potential hallucination
        ),
    }
    
    # Find matching response based on keywords
    for keyword, response in responses.items():
        if keyword.lower() in prompt.lower():
            # Add some randomized "hallucinations" to demonstrate verification
            if random.random() < 0.3:
                if keyword == "Einstein":
                    response += " He also invented the telephone."  # Hallucination
                elif keyword == "photosynthesis":
                    response += " The process was first discovered by Einstein."
            return {"response": response}
            
    return {"response": "I don't have enough information to answer that question."}

def process_query_with_graph(query: str, 
                            entity_extractor: EntityExtractorOperator,
                            fact_verifier: FactVerifierOperator) -> Dict:
    """Process a query using Ember's XCS execution engine."""
    
    # This simplifies the example while we debug the issues
    extracted_entities = entity_extractor(inputs=EntityExtractionInput(text=query)).entities
    llm_response = simulate_llm_response({"text": query})
    verification_results = fact_verifier(llm_response)
    
    return {
        "entities": extracted_entities,
        "response": llm_response["response"],
        "verification_results": verification_results
    }

def main():
    """Run the knowledge graph example."""
    print("Ember Knowledge Graph Example")
    print("============================")
    
    knowledge_graph = create_mock_knowledge_graph()
    entity_extractor = EntityExtractorOperator()
    fact_verifier = FactVerifierOperator(knowledge_graph)
    
    queries = [
        "Tell me about Einstein's contributions to physics.",
        "What is the relationship between photosynthesis and plants?",
        "Describe the structure of DNA.",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        result = process_query_with_graph(
            query=query,
            entity_extractor=entity_extractor,
            fact_verifier=fact_verifier
        )
        
        # 1. Extract relevant context from knowledge graph
        extracted_entities = result["entities"]
        print("\nExtracted Entities:")
        for entity in extracted_entities:
            print(f"- {entity.name} ({entity.type})")
        
        # 2. Generate response with context
        response = result["response"]
        print(f"\nGenerated Response:\n{response}")
        
        # 3. Verify response against knowledge graph
        verification_results = result["verification_results"]
        
        # 4. Display verification results
        print("\nVerification Results:")
        if verification_results:
            for statement, confidence in verification_results.items():
                status = "✓" if confidence > 0.8 else "?"
                print(f"{status} {statement}: {confidence:.2f} confidence")
        else:
            print("No verifiable statements found in response.")
            
    print("\nNote: This example uses simulated responses and verification.")
    print("In a real Ember pipeline, these would use actual language models. This file can be modified to include LLM calls with the knowledge graph implementation")

if __name__ == "__main__":
    main()
