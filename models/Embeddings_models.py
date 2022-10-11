import torch
from torch.nn import Module, Embedding
from torch.nn import functional as F

class KGEModel(Module):
    # Construct embeddings abstract class
    def __init__(self, KG) -> None:
        super().__init__()
        self.KG = KG
        return None
    
    def train():
        return None

class TransE(KGEModel):
    def __init__(self, KG, embedding_dim, device):
        """TransE Construction

        Args:
            KG (KnowledgeGraph): Knowledge graph embedded
            embedding_dim (int): number of dimensions for the embeddings
            device (string): device for the calculation
        """
        KGEModel.__init__(self, KG)
        self.embedding_dim = embedding_dim
        
        len_ent = self.KG.__len_entities__()
        len_rel = self.KG.__len_relations__()
        
        initrange = 6 / (self.embedding_dim ** 0.5)
        
        # Construction of embeddings for entities
        self.ent_embeddings = Embedding(
            num_embeddings=len_ent,
            embedding_dim=self.embedding_dim
        ).to(device)
        self.ent_embeddings.weight.data.uniform_(-initrange, initrange)
        
        # Construction of embeddings for relations
        self.rel_embeddings = Embedding(
            num_embeddings=len_rel,
            embedding_dim=self.embedding_dim
        ).to(device)
        self.rel_embeddings.weight.data.uniform_(-initrange, initrange)
        self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)  

    def get_entities_embeddings(self, entities_list_ids):
        """get list of entites embeddings by the index list of entities
        
        Args:
            entities_list_ids (List): List of the indexes of embeddings asked
        
        Returns:
            Tensor: List of embeddings
        """
        return torch.index_select(
            self.ent_embeddings.weight,
            0,
            entities_list_ids
        )
    
    def get_relations_embeddings(self, relations_list_ids):
        """get list of relations embeddings by the index list of entities
        
        Args:
            entities_list_ids (List): List of the indexes of embeddings asked
        
        Returns:
            Tensor: List of embeddings
        """
        return torch.index_select(
            self.ent_embeddings.weight,
            0,
            relations_list_ids
        )
    
    ### Problème du à une différence de taille sur le dernier batch
    def score(self, h, r, t):
        return torch.sum(h + r - t, dim=1)

    def batch_score(self, ids_true_batch, ids_false_batch):
        """Calculation of criterion for batch of true triplet and negative triplet

        Args:
            ids_true_batch (Tensor): batch of existing triplets described by ids
            ids_false_batch (Tensir): batch of negative generated triplets described by ids

        Returns:
            true_score :
            head_false_score :
            tail_false_score :
        """
        with torch.no_grad():
            self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        
        ids_true_h_batch = ids_true_batch[0]
        ids_true_r_batch = ids_true_batch[1]
        ids_true_t_batch = ids_true_batch[2]
        
        h_true = self.get_entities_embeddings(ids_true_h_batch)
        r_true = self.get_relations_embeddings(ids_true_r_batch)
        t_true = self.get_entities_embeddings(ids_true_t_batch)
        
        ids_false_h_batch = ids_false_batch[0]
        ids_false_t_batch = ids_false_batch[1]
        
        h_false = self.get_entities_embeddings(ids_false_h_batch)
        t_false = self.get_entities_embeddings(ids_false_t_batch)
        
        true_score = self.score(h_true, r_true, t_true)
        head_false_score = self.score(h_false, r_true, t_true)
        tail_false_score = self.score(h_true, r_true, t_false)
        
        return true_score, head_false_score, tail_false_score