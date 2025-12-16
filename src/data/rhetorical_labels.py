"""
Rhetorical Role Label Generation

Derives rhetorical role labels from existing BRAT annotations.
These labels can be used to train a rhetorical role classifier.

Rhetorical Roles:
- THESIS: Main position statement (MajorClaims)
- MAIN_ARGUMENT: Primary supporting claims (For Claims supporting MajorClaim)
- COUNTER_ARGUMENT: Opposing views (Against Claims)
- REBUTTAL: Response to counter-arguments (For Claims that attack Against Claims)
- EVIDENCE: Supporting premises (Premises)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.brat_parser import Essay, Component, Relation


class RhetoricalRole(Enum):
    """Rhetorical roles in argument structure."""
    THESIS = 0
    MAIN_ARGUMENT = 1
    COUNTER_ARGUMENT = 2
    REBUTTAL = 3
    EVIDENCE = 4
    
    @classmethod
    def from_int(cls, value: int) -> 'RhetoricalRole':
        for role in cls:
            if role.value == value:
                return role
        raise ValueError(f"Invalid role value: {value}")
    
    @classmethod
    def names(cls) -> List[str]:
        return [role.name for role in cls]


@dataclass
class LabeledComponent:
    """A component with its derived rhetorical role."""
    component: Component
    role: RhetoricalRole
    essay_id: str
    prompt: Optional[str] = None
    
    @property
    def text(self) -> str:
        return self.component.text
    
    @property
    def text_with_context(self) -> str:
        """Text with essay prompt as context."""
        if self.prompt:
            return f"{self.prompt} [SEP] {self.component.text}"
        return self.component.text


class RhetoricalLabelGenerator:
    """
    Generate rhetorical role labels from BRAT annotations.
    
    Label Derivation Rules:
    1. MajorClaim → THESIS
    2. Premise → EVIDENCE
    3. Claim with stance=Against → COUNTER_ARGUMENT
    4. Claim with stance=For that attacks an Against claim → REBUTTAL
    5. Other For Claims → MAIN_ARGUMENT
    """
    
    def __init__(self):
        self.stats = Counter()
    
    def _find_attack_targets(self, essay: Essay) -> Dict[str, List[str]]:
        """
        Build a mapping of component_id -> list of components that attack it.
        
        Returns:
            Dict mapping target_id to list of source_ids that attack it
        """
        attacks_on = {}
        for rel in essay.attack_relations:
            if rel.target_id not in attacks_on:
                attacks_on[rel.target_id] = []
            attacks_on[rel.target_id].append(rel.source_id)
        return attacks_on
    
    def _find_support_targets(self, essay: Essay) -> Dict[str, str]:
        """
        Build a mapping of component_id -> what it directly supports.
        
        Returns:
            Dict mapping source_id to target_id it supports
        """
        supports = {}
        for rel in essay.support_relations:
            supports[rel.source_id] = rel.target_id
        return supports
    
    def _identify_rebuttals(self, essay: Essay) -> set:
        """
        Find components that attack Against claims (rebuttals).
        
        In this dataset, rebuttals are typically Premises that attack 
        counter-arguments (Against claims). This is "rebuttal evidence" -
        evidence provided to counter the counter-argument.
        """
        attacks_on = self._find_attack_targets(essay)
        rebuttal_ids = set()
        
        # For each Against claim, check if any components attack it
        for claim in essay.against_claims:
            if claim.id in attacks_on:
                for attacker_id in attacks_on[claim.id]:
                    attacker = essay.components.get(attacker_id)
                    # Premises attacking Against claims are rebuttals
                    if attacker and attacker.type == 'Premise':
                        rebuttal_ids.add(attacker_id)
                    # For claims attacking Against claims are also rebuttals
                    elif attacker and attacker.type == 'Claim' and attacker.stance == 'For':
                        rebuttal_ids.add(attacker_id)
        
        return rebuttal_ids
    
    def _identify_main_arguments(self, essay: Essay) -> set:
        """
        Find For claims that directly support MajorClaims.
        
        These are the primary supporting arguments.
        """
        supports = self._find_support_targets(essay)
        main_arg_ids = set()
        
        major_claim_ids = {mc.id for mc in essay.major_claims}
        
        for claim in essay.for_claims:
            # Check if this claim directly supports a MajorClaim
            if claim.id in supports:
                target_id = supports[claim.id]
                if target_id in major_claim_ids:
                    main_arg_ids.add(claim.id)
        
        return main_arg_ids
    
    def generate_labels(self, essay: Essay) -> List[LabeledComponent]:
        """
        Generate rhetorical role labels for all components in an essay.
        
        Args:
            essay: Essay object with annotations
            
        Returns:
            List of LabeledComponent objects
        """
        labeled = []
        
        # Pre-compute special categories
        rebuttal_ids = self._identify_rebuttals(essay)
        main_arg_ids = self._identify_main_arguments(essay)
        
        for comp_id, component in essay.components.items():
            role = self._determine_role(component, rebuttal_ids, main_arg_ids)
            
            labeled.append(LabeledComponent(
                component=component,
                role=role,
                essay_id=essay.essay_id,
                prompt=essay.prompt
            ))
            
            self.stats[role.name] += 1
        
        return labeled
    
    def _determine_role(
        self, 
        component: Component, 
        rebuttal_ids: set,
        main_arg_ids: set
    ) -> RhetoricalRole:
        """Determine the rhetorical role of a component."""
        
        # Rule 1: MajorClaim → THESIS
        if component.type == 'MajorClaim':
            return RhetoricalRole.THESIS
        
        # Rule 2: Check for REBUTTAL first (before EVIDENCE)
        # Components that attack Against claims are rebuttals
        if component.id in rebuttal_ids:
            return RhetoricalRole.REBUTTAL
        
        # Rule 3: Premise → EVIDENCE (if not a rebuttal)
        if component.type == 'Premise':
            return RhetoricalRole.EVIDENCE
        
        # Rules for Claims
        if component.type == 'Claim':
            # Rule 4: Against claim → COUNTER_ARGUMENT
            if component.stance == 'Against':
                return RhetoricalRole.COUNTER_ARGUMENT
            
            # Rule 5: For claim → MAIN_ARGUMENT
            return RhetoricalRole.MAIN_ARGUMENT
        
        # Fallback (shouldn't reach here)
        return RhetoricalRole.EVIDENCE
    
    def generate_dataset(
        self, 
        essays: List[Essay]
    ) -> Tuple[List[str], List[int], List[str]]:
        """
        Generate training dataset from multiple essays.
        
        Args:
            essays: List of Essay objects
            
        Returns:
            Tuple of (texts, labels, essay_ids)
        """
        texts = []
        labels = []
        essay_ids = []
        
        for essay in essays:
            labeled_components = self.generate_labels(essay)
            for lc in labeled_components:
                texts.append(lc.text_with_context)
                labels.append(lc.role.value)
                essay_ids.append(lc.essay_id)
        
        return texts, labels, essay_ids
    
    def print_statistics(self):
        """Print label distribution statistics."""
        print("\n=== Rhetorical Role Distribution ===")
        total = sum(self.stats.values())
        for role in RhetoricalRole:
            count = self.stats[role.name]
            pct = 100 * count / total if total > 0 else 0
            print(f"  {role.name:20s}: {count:5d} ({pct:5.1f}%)")
        print(f"  {'TOTAL':20s}: {total:5d}")


def main():
    """Test label generation."""
    from src.data.dataset_builder import DatasetBuilder
    
    print("Loading dataset...")
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    # Generate labels
    generator = RhetoricalLabelGenerator()
    
    print("\nGenerating rhetorical labels...")
    all_essays = builder.get_all_essays()
    texts, labels, essay_ids = generator.generate_dataset(all_essays)
    
    generator.print_statistics()
    
    # Show some examples
    print("\n=== Sample Labels ===")
    for role in RhetoricalRole:
        print(f"\n{role.name}:")
        for i, (text, label, eid) in enumerate(zip(texts, labels, essay_ids)):
            if label == role.value:
                display_text = text.split('[SEP]')[-1].strip()[:80]
                print(f"  [{eid}] \"{display_text}...\"")
                if i > 2:
                    break


if __name__ == "__main__":
    main()

