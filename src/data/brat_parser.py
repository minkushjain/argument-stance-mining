"""
BRAT Annotation Parser for Argument Annotated Essays Corpus (AAEC)

Parses .ann files containing:
- T-lines: Components (MajorClaim, Claim, Premise) with character spans
- A-lines: Stance annotations (For/Against) linked to Claims
- R-lines: Relations (supports/attacks) between components
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class Component:
    """Represents an argumentative component (MajorClaim, Claim, or Premise)"""
    id: str
    type: str  # MajorClaim, Claim, or Premise
    start: int
    end: int
    text: str
    stance: Optional[str] = None  # For, Against, or None (only Claims have stance)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'start': self.start,
            'end': self.end,
            'text': self.text,
            'stance': self.stance
        }


@dataclass
class Relation:
    """Represents a relation between components (supports or attacks)"""
    id: str
    type: str  # supports or attacks
    source_id: str  # Arg1
    target_id: str  # Arg2
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'source_id': self.source_id,
            'target_id': self.target_id
        }


@dataclass
class Essay:
    """Represents a fully parsed essay with all annotations"""
    essay_id: str
    text: str
    prompt: Optional[str] = None
    split: Optional[str] = None  # TRAIN or TEST
    components: Dict[str, Component] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'essay_id': self.essay_id,
            'text': self.text,
            'prompt': self.prompt,
            'split': self.split,
            'components': {k: v.to_dict() for k, v in self.components.items()},
            'relations': [r.to_dict() for r in self.relations]
        }
    
    @property
    def major_claims(self) -> List[Component]:
        return [c for c in self.components.values() if c.type == 'MajorClaim']
    
    @property
    def claims(self) -> List[Component]:
        return [c for c in self.components.values() if c.type == 'Claim']
    
    @property
    def premises(self) -> List[Component]:
        return [c for c in self.components.values() if c.type == 'Premise']
    
    @property
    def for_claims(self) -> List[Component]:
        return [c for c in self.claims if c.stance == 'For']
    
    @property
    def against_claims(self) -> List[Component]:
        return [c for c in self.claims if c.stance == 'Against']
    
    @property
    def support_relations(self) -> List[Relation]:
        return [r for r in self.relations if r.type == 'supports']
    
    @property
    def attack_relations(self) -> List[Relation]:
        return [r for r in self.relations if r.type == 'attacks']


class BratParser:
    """Parser for BRAT annotation format used in AAEC dataset"""
    
    # Regex patterns for parsing .ann files
    COMPONENT_PATTERN = re.compile(
        r'^(T\d+)\t(MajorClaim|Claim|Premise)\s+(\d+)\s+(\d+)\t(.+)$'
    )
    STANCE_PATTERN = re.compile(
        r'^(A\d+)\tStance\s+(T\d+)\s+(For|Against)$'
    )
    RELATION_PATTERN = re.compile(
        r'^(R\d+)\t(supports|attacks)\s+Arg1:(T\d+)\s+Arg2:(T\d+)'
    )
    
    def __init__(self, brat_dir: str):
        """
        Initialize the parser with the BRAT project directory.
        
        Args:
            brat_dir: Path to the directory containing .ann and .txt files
        """
        self.brat_dir = Path(brat_dir)
        if not self.brat_dir.exists():
            raise FileNotFoundError(f"BRAT directory not found: {brat_dir}")
    
    def parse_annotation_file(self, ann_path: Path) -> Tuple[Dict[str, Component], List[Relation]]:
        """
        Parse a single .ann file.
        
        Args:
            ann_path: Path to the .ann file
            
        Returns:
            Tuple of (components dict, relations list)
        """
        components: Dict[str, Component] = {}
        relations: List[Relation] = []
        stance_annotations: Dict[str, str] = {}  # component_id -> stance
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Try to match component (T-line)
                comp_match = self.COMPONENT_PATTERN.match(line)
                if comp_match:
                    comp_id, comp_type, start, end, text = comp_match.groups()
                    components[comp_id] = Component(
                        id=comp_id,
                        type=comp_type,
                        start=int(start),
                        end=int(end),
                        text=text
                    )
                    continue
                
                # Try to match stance annotation (A-line)
                stance_match = self.STANCE_PATTERN.match(line)
                if stance_match:
                    _, target_id, stance = stance_match.groups()
                    stance_annotations[target_id] = stance
                    continue
                
                # Try to match relation (R-line)
                rel_match = self.RELATION_PATTERN.match(line)
                if rel_match:
                    rel_id, rel_type, source_id, target_id = rel_match.groups()
                    relations.append(Relation(
                        id=rel_id,
                        type=rel_type,
                        source_id=source_id,
                        target_id=target_id
                    ))
                    continue
        
        # Apply stance annotations to components
        for comp_id, stance in stance_annotations.items():
            if comp_id in components:
                components[comp_id].stance = stance
        
        return components, relations
    
    def parse_text_file(self, txt_path: Path) -> str:
        """Read essay text from .txt file."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse_essay(self, essay_id: str) -> Optional[Essay]:
        """
        Parse a single essay by its ID.
        
        Args:
            essay_id: Essay identifier (e.g., 'essay001')
            
        Returns:
            Essay object or None if files not found
        """
        ann_path = self.brat_dir / f"{essay_id}.ann"
        txt_path = self.brat_dir / f"{essay_id}.txt"
        
        if not ann_path.exists() or not txt_path.exists():
            return None
        
        text = self.parse_text_file(txt_path)
        components, relations = self.parse_annotation_file(ann_path)
        
        return Essay(
            essay_id=essay_id,
            text=text,
            components=components,
            relations=relations
        )
    
    def parse_all_essays(self) -> List[Essay]:
        """
        Parse all essays in the BRAT directory.
        
        Returns:
            List of Essay objects
        """
        essays = []
        
        # Find all .ann files
        ann_files = sorted(self.brat_dir.glob("essay*.ann"))
        
        for ann_path in ann_files:
            essay_id = ann_path.stem
            essay = self.parse_essay(essay_id)
            if essay:
                essays.append(essay)
        
        return essays
    
    def get_statistics(self, essays: List[Essay]) -> Dict:
        """
        Compute statistics over parsed essays.
        
        Args:
            essays: List of Essay objects
            
        Returns:
            Dictionary with various statistics
        """
        stats = {
            'total_essays': len(essays),
            'total_components': 0,
            'major_claims': 0,
            'claims': 0,
            'premises': 0,
            'stance_for': 0,
            'stance_against': 0,
            'total_relations': 0,
            'supports': 0,
            'attacks': 0,
        }
        
        for essay in essays:
            stats['total_components'] += len(essay.components)
            stats['major_claims'] += len(essay.major_claims)
            stats['claims'] += len(essay.claims)
            stats['premises'] += len(essay.premises)
            stats['stance_for'] += len(essay.for_claims)
            stats['stance_against'] += len(essay.against_claims)
            stats['total_relations'] += len(essay.relations)
            stats['supports'] += len(essay.support_relations)
            stats['attacks'] += len(essay.attack_relations)
        
        # Compute ratios
        if stats['claims'] > 0:
            stats['for_ratio'] = stats['stance_for'] / stats['claims']
            stats['against_ratio'] = stats['stance_against'] / stats['claims']
        
        if stats['total_relations'] > 0:
            stats['support_ratio'] = stats['supports'] / stats['total_relations']
            stats['attack_ratio'] = stats['attacks'] / stats['total_relations']
        
        return stats


def main():
    """Test the BRAT parser."""
    import sys
    
    # Default path to extracted BRAT files
    brat_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0" / "extracted_brat" / "brat-project-final"
    
    if not brat_dir.exists():
        print(f"BRAT directory not found: {brat_dir}")
        print("Please extract brat-project-final.zip first.")
        sys.exit(1)
    
    print(f"Parsing BRAT files from: {brat_dir}")
    parser = BratParser(str(brat_dir))
    
    # Parse all essays
    essays = parser.parse_all_essays()
    print(f"\nParsed {len(essays)} essays")
    
    # Print statistics
    stats = parser.get_statistics(essays)
    print("\n=== Dataset Statistics ===")
    print(f"Total essays: {stats['total_essays']}")
    print(f"Total components: {stats['total_components']}")
    print(f"  - MajorClaims: {stats['major_claims']}")
    print(f"  - Claims: {stats['claims']}")
    print(f"  - Premises: {stats['premises']}")
    print(f"\nStance distribution:")
    print(f"  - For: {stats['stance_for']} ({stats.get('for_ratio', 0):.1%})")
    print(f"  - Against: {stats['stance_against']} ({stats.get('against_ratio', 0):.1%})")
    print(f"\nRelations: {stats['total_relations']}")
    print(f"  - Supports: {stats['supports']} ({stats.get('support_ratio', 0):.1%})")
    print(f"  - Attacks: {stats['attacks']} ({stats.get('attack_ratio', 0):.1%})")
    
    # Show sample essay
    if essays:
        sample = essays[0]
        print(f"\n=== Sample Essay: {sample.essay_id} ===")
        print(f"Text length: {len(sample.text)} chars")
        print(f"Components: {len(sample.components)}")
        print(f"  - MajorClaims: {len(sample.major_claims)}")
        print(f"  - Claims: {len(sample.claims)} (For: {len(sample.for_claims)}, Against: {len(sample.against_claims)})")
        print(f"  - Premises: {len(sample.premises)}")
        print(f"Relations: {len(sample.relations)}")


if __name__ == "__main__":
    main()

