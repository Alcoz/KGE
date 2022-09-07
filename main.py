from src.KnowledgeGraphDataset import KnowledgeGraphDataset

FILE = "data/FB15K-237.2/Release/train.txt"

# Création du Dataset pour les graphes
dataset = KnowledgeGraphDataset(FILE)

# Création des Embeddings