import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
from typing import ClassVar
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class SuperpositionDataset(Dataset):
    def __init__(self, model, num_samples: int):
        self.model = model
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.model.generate_data(1).squeeze(0)

class SuperpositionTransformer(nn.Module):
    feature_probability: ClassVar[torch.Tensor] = None
    importance: ClassVar[torch.Tensor] = None
    
    @classmethod
    def set_feature_probability(cls, value: torch.Tensor):
        cls.feature_probability = value.to(cls.feature_probability.device if cls.feature_probability is not None else 'cpu')
    
    @classmethod
    def set_importance(cls, value: torch.Tensor):
        cls.importance = value.to(cls.importance.device if cls.importance is not None else 'cpu')

    def __init__(self, num_features: int, hidden_size: int, num_instances: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_instances = num_instances
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = GPT2Config(
            n_positions=32,
            n_embd=hidden_size,
            n_layer=4,
            n_head=4,
            n_inner=hidden_size*4
        )
        
        self.input_projection = nn.Linear(num_features, hidden_size)
        self.transformer = GPT2Model(self.config).to(self.device)
        self.output_projection = nn.Linear(hidden_size, num_features)
        
        if self.__class__.feature_probability is None:
            self.set_feature_probability((20 ** -torch.linspace(0, 1, num_instances))[:, None])
        if self.__class__.importance is None:
            self.set_importance((0.9**torch.arange(num_features))[None, :])

    def forward(self, features):
        batch_size = features.shape[0]
        
        # Reshape features: [batch, instances, features] -> [batch*instances, features]
        features_flat = features.view(-1, self.num_features)
        
        # Project to hidden size
        hidden = self.input_projection(features_flat)
        hidden = hidden.unsqueeze(1)  # Add sequence length dimension
        
        # Pass through transformer
        transformer_output = self.transformer(inputs_embeds=hidden).last_hidden_state
        transformer_output = transformer_output.squeeze(1)  # Remove sequence length dimension
        
        # Project back to feature size
        output = self.output_projection(transformer_output)
        
        # Reshape back: [batch*instances, features] -> [batch, instances, features]
        output = output.view(batch_size, self.num_instances, self.num_features)
        
        return F.relu(output)

    def generate_data(self, num_batch):
        feature = torch.rand((num_batch, self.num_instances, self.num_features), device=self.device)
        batch = torch.where(
            torch.rand((num_batch, self.num_instances, self.num_features), device=self.device) <= self.feature_probability,
            feature,
            torch.zeros((), device=self.device)
        )
        return batch

    def visualize_embeddings(self, writer: SummaryWriter, step: int):
        with torch.no_grad():
            weight = self.input_projection.weight.detach().cpu().numpy()
            
            writer.add_embedding(
                weight,
                metadata=[f'Hidden_{i}' for i in range(self.hidden_size)],
                tag=f'raw_embeddings',
                global_step=step
            )
            
            writer.add_histogram('weights/input_projection', weight, step)
            writer.add_histogram('weights/output_projection', 
                               self.output_projection.weight.detach().cpu().numpy(), step)

def optimize(model, config_dict):
    writer = SummaryWriter(f'runs/transformer_superposition')
    
    num_batch = config_dict['batch_size']
    num_steps = config_dict['num_steps']
    lr = config_dict['learning_rate']
    viz_interval = config_dict.get('visualization_interval', 100)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    with tqdm(range(num_steps), desc="Training", ncols=100) as t:
        running_loss = 0.0
        for step in t:
            opt.zero_grad()
            batch = model.generate_data(num_batch)
            out = model(batch)
            error = model.importance * (batch.abs() - out)**2
            loss = torch.mean(error)
            loss.backward()
            opt.step()
            
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            t.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss:.4f}'
            })
            
            writer.add_scalar('Loss/train', loss.item(), step)
            
            if step % viz_interval == 0:
                model.visualize_embeddings(writer, step)
    
    writer.close()

def main():
    config = {
        'num_features': 128,
        'hidden_size': 64,
        'num_instances': 10,
        'batch_size': 32,
        'num_steps': 1000,
        'learning_rate': 1e-4,
        'visualization_interval': 100
    }
    
    model = SuperpositionTransformer(
        num_features=config['num_features'],
        hidden_size=config['hidden_size'],
        num_instances=config['num_instances']
    )
    
    optimize(model, config)

if __name__ == "__main__":
    main()