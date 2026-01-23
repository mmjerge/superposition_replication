import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Model
from typing import ClassVar
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import collections as mc
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO

class SuperpositionTransformer(nn.Module):
    feature_probability: ClassVar[torch.Tensor] = None
    importance: ClassVar[torch.Tensor] = None
    
    def __init__(self, num_features: int, hidden_size: int, num_instances: int):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_instances = num_instances
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.__class__.feature_probability is None:
            self.__class__.feature_probability = (20 ** -torch.linspace(0, 1, num_instances))[:, None].to(self.device)
        if self.__class__.importance is None:
            self.__class__.importance = (0.9**torch.arange(num_features))[None, :].to(self.device)

        self.config = GPT2Config(
            n_positions=32,
            n_embd=hidden_size,
            n_layer=4,
            n_head=4,
            n_inner=hidden_size*4
        )
        
        self.input_projection = nn.Linear(num_features, hidden_size)
        self.transformer = GPT2Model(self.config)
        self.output_projection = nn.Linear(hidden_size, num_features)
        
        self.to(self.device)

    def forward(self, features):
        batch_size = features.shape[0]
        features_flat = features.view(-1, self.num_features)
        hidden = self.input_projection(features_flat)
        hidden = hidden.unsqueeze(1)
        transformer_output = self.transformer(inputs_embeds=hidden).last_hidden_state
        transformer_output = transformer_output.squeeze(1)
        output = self.output_projection(transformer_output)
        output = output.view(batch_size, self.num_instances, self.num_features)
        return F.relu(output)

    def generate_data(self, num_batch):
        feature = torch.rand((num_batch, self.num_instances, self.num_features), device=self.device)
        feature_prob = self.__class__.feature_probability.to(self.device)
        batch = torch.where(
            torch.rand((num_batch, self.num_instances, self.num_features), device=self.device) <= feature_prob,
            feature,
            torch.zeros((), device=self.device)
        )
        return batch

class SuperpositionVisualizer:
    def __init__(self, model):
        self.model = model
        self.writer = SummaryWriter('runs/superposition_experiment')
        print(f"Initializing visualizer. Model is on device: {next(model.parameters()).device}")

    def plot_weight_vectors(self, step):
        """Create vector plot showing superposition in 2D space with enhanced visibility"""
        with torch.no_grad():
            weights = self.model.input_projection.weight.detach().cpu().numpy()
            weights_2d = weights[:, :2]
            
            print(f"\nStep {step} - Weight stats:")
            print(f"Weight shape: {weights_2d.shape}")
            print(f"Weight range: [{weights_2d.min():.4f}, {weights_2d.max():.4f}]")
            
            max_magnitude = np.abs(weights_2d).max()
            if max_magnitude > 0:
                weights_2d = weights_2d * (1.0 / max_magnitude)
                print(f"Normalized weight range: [{weights_2d.min():.4f}, {weights_2d.max():.4f}]")
            
            importance = self.model.importance[0].cpu().numpy()

        num_instances = self.model.num_instances
        fig, axs = plt.subplots(1, num_instances, figsize=(2*num_instances, 2))
        
        if num_instances == 1:
            axs = [axs]
        
        num_vectors = len(weights_2d)
        colors = plt.cm.YlGn(np.linspace(0.3, 0.9, num_vectors))
        
        for idx, ax in enumerate(axs):
            for i, (wx, wy) in enumerate(weights_2d):
                line = ax.plot([0, wx], [0, wy], '-', color=colors[i], 
                             linewidth=2, alpha=0.7)
                point = ax.scatter(wx, wy, color=colors[i], s=100, 
                                 alpha=0.8, zorder=3)
                
            self._setup_plot_style(ax)
            
            ax.grid(True, linestyle='--', alpha=0.3)
            
            ax.set_title(f'Instance {idx+1}', pad=10)
        
        plt.tight_layout()
        
        plt.savefig(f'vector_plot_step_{step}.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    pad_inches=0.1)
        
        self.writer.add_figure('weight_vectors', fig, step)
        
        plt.close()

    def _setup_plot_style(self, ax):
        """Enhanced plot styling"""
        ax.set_facecolor('white')
        
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        ax.set_aspect('equal')
        limit = 1.2  
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        ax.set_xticks([-1, -0.5, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0.5, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    def log_training_metrics(self, loss, step):
        self.writer.add_scalar('Loss/train', loss, step)
    
    def close(self):
        self.writer.close()

def optimize_with_visualization(model, config_dict):
    print(f"Starting optimization. Model is on device: {next(model.parameters()).device}")
    visualizer = SuperpositionVisualizer(model)
    
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
            
            visualizer.log_training_metrics(loss.item(), step)
            
            if step % viz_interval == 0:
                visualizer.plot_weight_vectors(step)
                
                with torch.no_grad():
                    weight_norm = torch.norm(model.input_projection.weight).item()
                    print(f"\nStep {step} - Weight norm: {weight_norm:.4f}")
    
    visualizer.close()

def main():
    config = {
        'num_features': 64,    
        'hidden_size': 32,     
        'num_instances': 8,
        'batch_size': 32,
        'num_steps': 1000,
        'learning_rate': 1e-4,
        'visualization_interval': 50
    }
    
    print("Creating model...")
    model = SuperpositionTransformer(
        num_features=config['num_features'],
        hidden_size=config['hidden_size'],
        num_instances=config['num_instances']
    )
    
    print(f"Model created. Device: {next(model.parameters()).device}")
    optimize_with_visualization(model, config)

if __name__ == "__main__":
    main()