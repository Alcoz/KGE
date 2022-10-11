from random import sample

import torch
from torch.utils.data import Dataset

from .KG_utils import triplet_initializer

class KnowledgeGraph(Dataset):
    def __init__(self, root) -> None:
        """
        Args:
            root (str): path trough the knowledge graph as txt file
        """
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
    
    def generate_false_entities_candidate(self, batch_size):
        """Dumb false entities generation candidate : the canditate is choosed randomly in the set

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        num_entities = self.__len_entities__()
        return [torch.tensor(sample(range(0, num_entities), batch_size)), torch.tensor(sample(range(0, num_entities), batch_size))]
