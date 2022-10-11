from src.KnowledgeGraphDataset import KnowledgeGraph
from models.Embeddings_models import TransE
from src.Experiment import Experiment
import torch

FILE = "data/FB15K-237.2/Release/train.txt"

# Hyperparameter
num_epochs = 25
batch_size = 32
margin = 1
norm = 2
learning_rate = 0.01
num_dimensions = 50

# Cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# KnowledgeGraph
KG = KnowledgeGraph(FILE)

# KGE Model
model = TransE(KG, num_dimensions, device)

# Experiment
experiment = Experiment(KG, 
                        model, 
                        num_epochs, 
                        batch_size, 
                        margin,
                        norm, 
                        learning_rate, 
                        num_dimensions, 
                        device)

# Run the experiment
experiment.run()