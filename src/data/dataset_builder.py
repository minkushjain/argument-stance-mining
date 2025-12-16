"""
Dataset Builder for Argument Annotated Essays Corpus (AAEC)

Merges BRAT annotations with essay texts, prompts, and train/test splits.
Provides unified dataset access for all downstream tasks.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import random

from .brat_parser import BratParser, Essay


@dataclass
class DatasetSplit:
    """Holds essay IDs for train/val/test splits"""
    train: List[str]
    val: List[str]
    test: List[str]


class DatasetBuilder:
    """
    Builds and manages the AAEC dataset with all metadata.
    
    Responsibilities:
    - Load BRAT annotations via BratParser
    - Load essay prompts from prompts.csv
    - Apply train/test split from train-test-split.csv
    - Create validation split from training data
    - Export processed data
    """
    
    def __init__(self, dataset_dir: str, val_ratio: float = 0.15, random_seed: int = 42):
        """
        Initialize the dataset builder.
        
        Args:
            dataset_dir: Path to ArgumentAnnotatedEssays-2.0 directory
            val_ratio: Ratio of training data to use for validation
            random_seed: Random seed for reproducibility
        """
        self.dataset_dir = Path(dataset_dir)
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        
        # Paths to various files
        self.brat_dir = self.dataset_dir / "extracted_brat" / "brat-project-final"
        self.prompts_path = self.dataset_dir / "prompts.csv"
        self.split_path = self.dataset_dir / "train-test-split.csv"
        
        # Data containers
        self.essays: Dict[str, Essay] = {}
        self.prompts: Dict[str, str] = {}
        self.official_splits: Dict[str, str] = {}  # essay_id -> TRAIN/TEST
        self.splits: Optional[DatasetSplit] = None
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that all required files/directories exist."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        if not self.brat_dir.exists():
            raise FileNotFoundError(f"BRAT directory not found: {self.brat_dir}. "
                                   "Please extract brat-project-final.zip first.")
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {self.prompts_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")
    
    def load_prompts(self) -> Dict[str, str]:
        """Load essay prompts from prompts.csv."""
        prompts = {}
        # Use latin-1 encoding as fallback for special characters in prompts
        with open(self.prompts_path, 'r', encoding='utf-8', errors='replace') as f:
            # File format: ESSAY;PROMPT (semicolon separated)
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    essay_file = row[0].strip()
                    prompt = row[1].strip()
                    # Convert essay001.txt to essay001
                    essay_id = essay_file.replace('.txt', '')
                    prompts[essay_id] = prompt
        
        self.prompts = prompts
        return prompts
    
    def load_official_splits(self) -> Dict[str, str]:
        """Load official train/test split from train-test-split.csv."""
        splits = {}
        with open(self.split_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    essay_id = row[0].strip().replace('"', '')
                    split = row[1].strip().replace('"', '')
                    splits[essay_id] = split
        
        self.official_splits = splits
        return splits
    
    def load_essays(self) -> Dict[str, Essay]:
        """Load all essays using BratParser."""
        parser = BratParser(str(self.brat_dir))
        essays_list = parser.parse_all_essays()
        
        # Convert to dict and add prompts/splits
        self.essays = {}
        for essay in essays_list:
            # Add prompt if available
            if essay.essay_id in self.prompts:
                essay.prompt = self.prompts[essay.essay_id]
            
            # Add official split
            if essay.essay_id in self.official_splits:
                essay.split = self.official_splits[essay.essay_id]
            
            self.essays[essay.essay_id] = essay
        
        return self.essays
    
    def create_splits(self) -> DatasetSplit:
        """
        Create train/val/test splits.
        
        Uses official TRAIN/TEST split, then creates validation set
        from a portion of training data.
        """
        random.seed(self.random_seed)
        
        # Separate by official split
        train_ids = [eid for eid, split in self.official_splits.items() if split == 'TRAIN']
        test_ids = [eid for eid, split in self.official_splits.items() if split == 'TEST']
        
        # Shuffle training IDs
        random.shuffle(train_ids)
        
        # Split training into train/val
        val_size = int(len(train_ids) * self.val_ratio)
        val_ids = train_ids[:val_size]
        train_ids = train_ids[val_size:]
        
        self.splits = DatasetSplit(
            train=sorted(train_ids),
            val=sorted(val_ids),
            test=sorted(test_ids)
        )
        
        return self.splits
    
    def build(self) -> 'DatasetBuilder':
        """
        Build the complete dataset.
        
        Loads all data and creates splits. Call this before accessing data.
        """
        print("Loading prompts...")
        self.load_prompts()
        print(f"  Loaded {len(self.prompts)} prompts")
        
        print("Loading official splits...")
        self.load_official_splits()
        train_count = sum(1 for s in self.official_splits.values() if s == 'TRAIN')
        test_count = sum(1 for s in self.official_splits.values() if s == 'TEST')
        print(f"  Official split: {train_count} TRAIN, {test_count} TEST")
        
        print("Loading essays...")
        self.load_essays()
        print(f"  Loaded {len(self.essays)} essays")
        
        print("Creating train/val/test splits...")
        self.create_splits()
        print(f"  Train: {len(self.splits.train)}, Val: {len(self.splits.val)}, Test: {len(self.splits.test)}")
        
        return self
    
    def get_essays_by_split(self, split: str) -> List[Essay]:
        """Get essays for a specific split (train/val/test)."""
        if self.splits is None:
            raise RuntimeError("Dataset not built. Call build() first.")
        
        if split == 'train':
            ids = self.splits.train
        elif split == 'val':
            ids = self.splits.val
        elif split == 'test':
            ids = self.splits.test
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")
        
        return [self.essays[eid] for eid in ids if eid in self.essays]
    
    def get_all_essays(self) -> List[Essay]:
        """Get all essays."""
        return list(self.essays.values())
    
    def export_to_json(self, output_path: str):
        """Export the complete dataset to JSON."""
        data = {
            'metadata': {
                'total_essays': len(self.essays),
                'splits': {
                    'train': len(self.splits.train) if self.splits else 0,
                    'val': len(self.splits.val) if self.splits else 0,
                    'test': len(self.splits.test) if self.splits else 0
                },
                'val_ratio': self.val_ratio,
                'random_seed': self.random_seed
            },
            'splits': asdict(self.splits) if self.splits else None,
            'essays': {eid: essay.to_dict() for eid, essay in self.essays.items()}
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported dataset to {output_path}")
    
    def get_statistics(self) -> Dict:
        """Compute comprehensive dataset statistics."""
        stats = {
            'total_essays': len(self.essays),
            'splits': {
                'train': len(self.splits.train) if self.splits else 0,
                'val': len(self.splits.val) if self.splits else 0,
                'test': len(self.splits.test) if self.splits else 0
            },
            'components': {
                'total': 0,
                'major_claims': 0,
                'claims': 0,
                'premises': 0
            },
            'stance': {
                'for': 0,
                'against': 0
            },
            'relations': {
                'total': 0,
                'supports': 0,
                'attacks': 0
            },
            'per_essay': {
                'avg_components': 0,
                'avg_claims': 0,
                'avg_premises': 0,
                'avg_relations': 0
            }
        }
        
        for essay in self.essays.values():
            stats['components']['total'] += len(essay.components)
            stats['components']['major_claims'] += len(essay.major_claims)
            stats['components']['claims'] += len(essay.claims)
            stats['components']['premises'] += len(essay.premises)
            stats['stance']['for'] += len(essay.for_claims)
            stats['stance']['against'] += len(essay.against_claims)
            stats['relations']['total'] += len(essay.relations)
            stats['relations']['supports'] += len(essay.support_relations)
            stats['relations']['attacks'] += len(essay.attack_relations)
        
        n = len(self.essays)
        if n > 0:
            stats['per_essay']['avg_components'] = stats['components']['total'] / n
            stats['per_essay']['avg_claims'] = stats['components']['claims'] / n
            stats['per_essay']['avg_premises'] = stats['components']['premises'] / n
            stats['per_essay']['avg_relations'] = stats['relations']['total'] / n
        
        return stats


def main():
    """Test the dataset builder."""
    from pathlib import Path
    
    dataset_dir = Path(__file__).parent.parent.parent / "dataset" / "ArgumentAnnotatedEssays-2.0"
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "essays_parsed.json"
    
    print(f"Building dataset from: {dataset_dir}")
    
    builder = DatasetBuilder(str(dataset_dir))
    builder.build()
    
    # Print statistics
    stats = builder.get_statistics()
    print("\n=== Dataset Statistics ===")
    print(f"Total essays: {stats['total_essays']}")
    print(f"Splits: Train={stats['splits']['train']}, Val={stats['splits']['val']}, Test={stats['splits']['test']}")
    print(f"\nComponents: {stats['components']['total']}")
    print(f"  - MajorClaims: {stats['components']['major_claims']}")
    print(f"  - Claims: {stats['components']['claims']}")
    print(f"  - Premises: {stats['components']['premises']}")
    print(f"\nStance distribution:")
    print(f"  - For: {stats['stance']['for']}")
    print(f"  - Against: {stats['stance']['against']}")
    print(f"\nRelations: {stats['relations']['total']}")
    print(f"  - Supports: {stats['relations']['supports']}")
    print(f"  - Attacks: {stats['relations']['attacks']}")
    print(f"\nPer-essay averages:")
    print(f"  - Components: {stats['per_essay']['avg_components']:.1f}")
    print(f"  - Claims: {stats['per_essay']['avg_claims']:.1f}")
    print(f"  - Premises: {stats['per_essay']['avg_premises']:.1f}")
    print(f"  - Relations: {stats['per_essay']['avg_relations']:.1f}")
    
    # Export to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.export_to_json(str(output_path))
    
    # Show sample
    train_essays = builder.get_essays_by_split('train')
    if train_essays:
        sample = train_essays[0]
        print(f"\n=== Sample Train Essay: {sample.essay_id} ===")
        print(f"Prompt: {sample.prompt[:100]}..." if sample.prompt else "No prompt")
        print(f"Text: {sample.text[:200]}...")


if __name__ == "__main__":
    main()

