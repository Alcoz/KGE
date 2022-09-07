import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets

from .KG_utils import triplet_initializer

class KnowledgeGraphDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        
        entities_to_ids, relations_to_ids, ids_to_entities, ids_to_relation, triplets = triplet_initializer(root)
        
        # Dictionary id_to_{entites, relation}
        self.entities_to_ids = entities_to_ids
        self.relations_to_ids = relations_to_ids
        
        self.ids_to_entities = ids_to_entities
        self.ids_to_relation = ids_to_relation
        
        # List of triplets by ids
        self.triplets = triplets
        
    def __len__(self):
        return len(self.triplets)
    
    def __len_entities__(self):
        return len(self.entities_to_ids)
    
    def __len_relations__(self):
        return len(self.relations_to_ids)
    
    def __getitem__(self, idx):
        return self.triplets[idx]