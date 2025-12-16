"""
Essay-Level Feature Extraction for Stance Structure Analysis

Extracts comprehensive features from annotated essays:
- Stance Features: for_count, against_count, for_ratio, stance_balance
- Structural Features: component counts, evidence density, relation ratios
- Graph Features: argument tree depth, width, branching factor
- Positional Features: stance distribution by essay sections
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import json
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.brat_parser import Essay, Component, Relation


@dataclass
class EssayFeatures:
    """Complete feature set for a single essay."""
    essay_id: str
    
    # Stance Features
    for_count: int = 0
    against_count: int = 0
    for_ratio: float = 0.0
    against_ratio: float = 0.0
    stance_balance: float = 0.0  # |for - against| / total
    
    # Component Counts
    total_components: int = 0
    major_claim_count: int = 0
    claim_count: int = 0
    premise_count: int = 0
    
    # Structural Features
    evidence_density: float = 0.0  # premises / claims
    claim_density: float = 0.0  # claims / total_components
    major_claim_ratio: float = 0.0  # major_claims / total_components
    
    # Relation Features
    total_relations: int = 0
    support_count: int = 0
    attack_count: int = 0
    support_ratio: float = 0.0
    attack_ratio: float = 0.0
    relations_per_component: float = 0.0
    
    # Graph Features
    tree_depth: int = 0
    tree_width: int = 0  # Max components at any level
    avg_branching_factor: float = 0.0
    avg_path_to_major_claim: float = 0.0
    num_root_components: int = 0  # Components with no incoming relations
    num_leaf_components: int = 0  # Components with no outgoing relations
    
    # Positional Features (essay divided into thirds)
    for_in_first_third: int = 0
    for_in_middle_third: int = 0
    for_in_last_third: int = 0
    against_in_first_third: int = 0
    against_in_middle_third: int = 0
    against_in_last_third: int = 0
    first_against_position: float = -1.0  # Normalized position (0-1), -1 if no against
    last_against_position: float = -1.0
    
    # Text Features
    essay_length: int = 0
    avg_component_length: float = 0.0
    avg_claim_length: float = 0.0
    avg_premise_length: float = 0.0
    
    # Derived Category (for classification)
    stance_category: str = ""  # mostly_for, balanced, mostly_against
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy array for ML models."""
        # Select numerical features only
        features = [
            self.for_count, self.against_count, self.for_ratio, self.against_ratio,
            self.stance_balance, self.total_components, self.major_claim_count,
            self.claim_count, self.premise_count, self.evidence_density,
            self.claim_density, self.major_claim_ratio, self.total_relations,
            self.support_count, self.attack_count, self.support_ratio,
            self.attack_ratio, self.relations_per_component, self.tree_depth,
            self.tree_width, self.avg_branching_factor, self.avg_path_to_major_claim,
            self.num_root_components, self.num_leaf_components,
            self.for_in_first_third, self.for_in_middle_third, self.for_in_last_third,
            self.against_in_first_third, self.against_in_middle_third, self.against_in_last_third,
            self.first_against_position if self.first_against_position >= 0 else 1.0,
            self.last_against_position if self.last_against_position >= 0 else 0.0,
            self.essay_length, self.avg_component_length, self.avg_claim_length,
            self.avg_premise_length
        ]
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get names of features in the feature vector."""
        return [
            'for_count', 'against_count', 'for_ratio', 'against_ratio',
            'stance_balance', 'total_components', 'major_claim_count',
            'claim_count', 'premise_count', 'evidence_density',
            'claim_density', 'major_claim_ratio', 'total_relations',
            'support_count', 'attack_count', 'support_ratio',
            'attack_ratio', 'relations_per_component', 'tree_depth',
            'tree_width', 'avg_branching_factor', 'avg_path_to_major_claim',
            'num_root_components', 'num_leaf_components',
            'for_in_first_third', 'for_in_middle_third', 'for_in_last_third',
            'against_in_first_third', 'against_in_middle_third', 'against_in_last_third',
            'first_against_position', 'last_against_position',
            'essay_length', 'avg_component_length', 'avg_claim_length',
            'avg_premise_length'
        ]


class EssayFeatureExtractor:
    """
    Extracts comprehensive features from essays for stance structure analysis.
    """
    
    def __init__(self, balanced_threshold: float = 0.20):
        """
        Initialize the feature extractor.
        
        Args:
            balanced_threshold: Threshold for classifying essays as balanced.
                               If |for_ratio - 0.5| < threshold, essay is balanced.
                               Using 0.20 means for_ratio between 0.30 and 0.70 is balanced.
        """
        self.balanced_threshold = balanced_threshold
    
    def extract_features(self, essay: Essay) -> EssayFeatures:
        """
        Extract all features from a single essay.
        
        Args:
            essay: Essay object with annotations
            
        Returns:
            EssayFeatures object with all extracted features
        """
        features = EssayFeatures(essay_id=essay.essay_id)
        
        # Extract each feature category
        self._extract_stance_features(essay, features)
        self._extract_component_features(essay, features)
        self._extract_relation_features(essay, features)
        self._extract_graph_features(essay, features)
        self._extract_positional_features(essay, features)
        self._extract_text_features(essay, features)
        self._compute_stance_category(features)
        
        return features
    
    def _extract_stance_features(self, essay: Essay, features: EssayFeatures):
        """Extract stance-related features."""
        features.for_count = len(essay.for_claims)
        features.against_count = len(essay.against_claims)
        
        total_stance = features.for_count + features.against_count
        if total_stance > 0:
            features.for_ratio = features.for_count / total_stance
            features.against_ratio = features.against_count / total_stance
            features.stance_balance = abs(features.for_count - features.against_count) / total_stance
    
    def _extract_component_features(self, essay: Essay, features: EssayFeatures):
        """Extract component count and density features."""
        features.total_components = len(essay.components)
        features.major_claim_count = len(essay.major_claims)
        features.claim_count = len(essay.claims)
        features.premise_count = len(essay.premises)
        
        if features.claim_count > 0:
            features.evidence_density = features.premise_count / features.claim_count
        
        if features.total_components > 0:
            features.claim_density = features.claim_count / features.total_components
            features.major_claim_ratio = features.major_claim_count / features.total_components
    
    def _extract_relation_features(self, essay: Essay, features: EssayFeatures):
        """Extract relation-related features."""
        features.total_relations = len(essay.relations)
        features.support_count = len(essay.support_relations)
        features.attack_count = len(essay.attack_relations)
        
        if features.total_relations > 0:
            features.support_ratio = features.support_count / features.total_relations
            features.attack_ratio = features.attack_count / features.total_relations
        
        if features.total_components > 0:
            features.relations_per_component = features.total_relations / features.total_components
    
    def _extract_graph_features(self, essay: Essay, features: EssayFeatures):
        """Extract argument graph structure features."""
        if not essay.components or not essay.relations:
            return
        
        # Build adjacency lists
        # outgoing[source] = list of targets
        # incoming[target] = list of sources
        outgoing: Dict[str, List[str]] = defaultdict(list)
        incoming: Dict[str, List[str]] = defaultdict(list)
        
        for rel in essay.relations:
            outgoing[rel.source_id].append(rel.target_id)
            incoming[rel.target_id].append(rel.source_id)
        
        # Root components (no incoming relations)
        root_ids = [cid for cid in essay.components if cid not in incoming]
        features.num_root_components = len(root_ids)
        
        # Leaf components (no outgoing relations)
        leaf_ids = [cid for cid in essay.components if cid not in outgoing]
        features.num_leaf_components = len(leaf_ids)
        
        # Compute tree depth using BFS from leaves to major claims
        major_claim_ids = set(c.id for c in essay.major_claims)
        
        # Calculate depth from each component to any major claim
        depths = self._compute_depths_to_targets(essay.components.keys(), outgoing, major_claim_ids)
        
        if depths:
            valid_depths = [d for d in depths.values() if d < float('inf')]
            if valid_depths:
                features.tree_depth = max(valid_depths)
                features.avg_path_to_major_claim = sum(valid_depths) / len(valid_depths)
            else:
                # No paths to major claims found - use max relations as proxy for depth
                features.tree_depth = max(len(list(outgoing.values())), 1) if outgoing else 1
        
        # Compute tree width (max components at any depth level)
        depth_counts = defaultdict(int)
        for cid, depth in depths.items():
            if depth < float('inf'):
                depth_counts[depth] += 1
        
        if depth_counts:
            features.tree_width = max(depth_counts.values())
        
        # Average branching factor
        branching_factors = [len(targets) for targets in outgoing.values() if targets]
        if branching_factors:
            features.avg_branching_factor = sum(branching_factors) / len(branching_factors)
    
    def _compute_depths_to_targets(
        self, 
        component_ids: List[str], 
        outgoing: Dict[str, List[str]], 
        target_ids: set
    ) -> Dict[str, int]:
        """Compute shortest path depth from each component to any target."""
        depths = {}
        
        for cid in component_ids:
            if cid in target_ids:
                depths[cid] = 0
            else:
                # BFS to find shortest path to any target
                depth = self._bfs_to_targets(cid, outgoing, target_ids)
                depths[cid] = depth
        
        return depths
    
    def _bfs_to_targets(
        self, 
        start_id: str, 
        outgoing: Dict[str, List[str]], 
        target_ids: set
    ) -> int:
        """BFS to find shortest path from start to any target."""
        from collections import deque
        
        visited = set()
        queue = deque([(start_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if current_id in target_ids:
                return depth
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            for next_id in outgoing.get(current_id, []):
                if next_id not in visited:
                    queue.append((next_id, depth + 1))
        
        return float('inf')  # No path found
    
    def _extract_positional_features(self, essay: Essay, features: EssayFeatures):
        """Extract positional features based on where stances appear in essay."""
        essay_length = len(essay.text)
        if essay_length == 0:
            return
        
        third_length = essay_length / 3
        
        against_positions = []
        
        for claim in essay.claims:
            # Determine which third the claim is in
            position = claim.start
            normalized_position = position / essay_length
            
            if position < third_length:
                section = 'first'
            elif position < 2 * third_length:
                section = 'middle'
            else:
                section = 'last'
            
            if claim.stance == 'For':
                if section == 'first':
                    features.for_in_first_third += 1
                elif section == 'middle':
                    features.for_in_middle_third += 1
                else:
                    features.for_in_last_third += 1
            elif claim.stance == 'Against':
                against_positions.append(normalized_position)
                if section == 'first':
                    features.against_in_first_third += 1
                elif section == 'middle':
                    features.against_in_middle_third += 1
                else:
                    features.against_in_last_third += 1
        
        # First and last Against positions
        if against_positions:
            features.first_against_position = min(against_positions)
            features.last_against_position = max(against_positions)
    
    def _extract_text_features(self, essay: Essay, features: EssayFeatures):
        """Extract basic text features."""
        features.essay_length = len(essay.text)
        
        if essay.components:
            component_lengths = [len(c.text) for c in essay.components.values()]
            features.avg_component_length = sum(component_lengths) / len(component_lengths)
        
        if essay.claims:
            claim_lengths = [len(c.text) for c in essay.claims]
            features.avg_claim_length = sum(claim_lengths) / len(claim_lengths)
        
        if essay.premises:
            premise_lengths = [len(p.text) for p in essay.premises]
            features.avg_premise_length = sum(premise_lengths) / len(premise_lengths)
    
    def _compute_stance_category(self, features: EssayFeatures):
        """Compute stance category based on for_ratio."""
        if features.for_count + features.against_count == 0:
            features.stance_category = "no_stance"
        elif abs(features.for_ratio - 0.5) < self.balanced_threshold:
            features.stance_category = "balanced"
        elif features.for_ratio > 0.5:
            features.stance_category = "mostly_for"
        else:
            features.stance_category = "mostly_against"
    
    def extract_all(self, essays: List[Essay]) -> List[EssayFeatures]:
        """Extract features from all essays."""
        return [self.extract_features(essay) for essay in essays]
    
    def to_dataframe(self, features_list: List[EssayFeatures]):
        """Convert features to pandas DataFrame."""
        import pandas as pd
        
        data = [f.to_dict() for f in features_list]
        return pd.DataFrame(data)
    
    def to_feature_matrix(self, features_list: List[EssayFeatures]) -> Tuple[np.ndarray, List[str]]:
        """Convert features to numpy matrix with feature names."""
        matrix = np.array([f.to_feature_vector() for f in features_list])
        names = EssayFeatures.feature_names()
        return matrix, names
    
    def save_features(self, features_list: List[EssayFeatures], output_path: str):
        """Save features to JSON file."""
        data = {
            'feature_names': EssayFeatures.feature_names(),
            'features': [f.to_dict() for f in features_list]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(features_list)} essay features to {output_path}")


def main():
    """Test feature extraction."""
    from src.data.dataset_builder import DatasetBuilder
    
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "essay_features.json"
    
    print("Loading dataset...")
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    print("\nExtracting features...")
    extractor = EssayFeatureExtractor()
    all_essays = builder.get_all_essays()
    features = extractor.extract_all(all_essays)
    
    print(f"\nExtracted features for {len(features)} essays")
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extractor.save_features(features, str(output_path))
    
    # Show statistics
    print("\n=== Feature Statistics ===")
    
    # Convert to DataFrame for easy stats
    df = extractor.to_dataframe(features)
    
    print(f"\nStance Category Distribution:")
    print(df['stance_category'].value_counts())
    
    print(f"\nKey Feature Stats:")
    key_features = ['for_ratio', 'evidence_density', 'attack_ratio', 'tree_depth', 'avg_path_to_major_claim']
    for feat in key_features:
        print(f"  {feat}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}, "
              f"min={df[feat].min():.3f}, max={df[feat].max():.3f}")
    
    # Show sample
    print(f"\n=== Sample Features: {features[0].essay_id} ===")
    sample = features[0]
    print(f"  Stance: For={sample.for_count}, Against={sample.against_count}, Ratio={sample.for_ratio:.2f}")
    print(f"  Components: Total={sample.total_components}, Claims={sample.claim_count}, Premises={sample.premise_count}")
    print(f"  Relations: Total={sample.total_relations}, Support={sample.support_count}, Attack={sample.attack_count}")
    print(f"  Graph: Depth={sample.tree_depth}, Width={sample.tree_width}, AvgBranching={sample.avg_branching_factor:.2f}")
    print(f"  Category: {sample.stance_category}")


if __name__ == "__main__":
    main()

