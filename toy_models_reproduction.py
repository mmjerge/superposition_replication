import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import yaml
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, ClassVar

class ToyDataset(Dataset):
    def __init__(self, model, num_samples: int):
        self.model = model
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.model.generate_data(1).squeeze(0)

class ToyModel(nn.Module):
    feature_probability: ClassVar[torch.Tensor] = None
    importance: ClassVar[torch.Tensor] = None

    @classmethod
    def set_feature_probability(cls, value: torch.Tensor):
        cls.feature_probability = value.to(cls.feature_probability.device if cls.feature_probability is not None else 'cpu')

    @classmethod
    def set_importance(cls, value: torch.Tensor):
        cls.importance = value.to(cls.importance.device if cls.importance is not None else 'cpu')

    def __init__(self, num_features, num_hidden, num_instances, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_instances = num_instances
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.W = nn.Parameter(torch.empty((self.num_instances, self.num_features, self.num_hidden), device=self.device))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(torch.zeros((self.num_instances, self.num_features), device=self.device))
        if self.__class__.feature_probability is None:
            self.set_feature_probability((20 ** -torch.linspace(0, 1, self.num_instances))[:, None])
        if self.__class__.importance is None:
            self.set_importance((0.9**torch.arange(self.num_features))[None, :])

    def forward(self, features):
        # features shape: (batch_size, num_instances, num_features)
        # self.W shape: (num_instances, num_features, num_hidden)
        # self.b_final shape: (num_instances, num_features)
        hidden = torch.matmul(features.unsqueeze(2), self.W.unsqueeze(0)).squeeze(2)
        W_transposed = self.W.transpose(1, 2)
        out = torch.matmul(hidden.unsqueeze(2), W_transposed.unsqueeze(0)).squeeze(2)
        out = out + self.b_final.unsqueeze(0)
        out = F.relu(out)
        return out

    def generate_data(self, num_batch):
        feature = torch.rand((num_batch, self.num_instances, self.num_features), device=self.W.device)
        batch = torch.where(
            torch.rand((num_batch, self.num_instances, self.num_features), device=self.W.device) <= self.feature_probability,
            feature,
            torch.zeros((), device=self.W.device),
        )
        return batch

def optimize(model, config, **kwargs):
    with open(config, 'r') as file:
        cfg = yaml.safe_load(file)['toy_model_config']
    cfg.update(kwargs)
    wandb.init(project=cfg['model_name'], config=cfg)
    
    num_batch = int(cfg['batch_size'])
    num_of_steps = int(cfg['num_of_steps'])
    initial_lr = float(cfg['learning_rate'])
    scheduler_type = cfg['scheduler_type']
    
    lr_schedulers = {
        'linear': lambda step: 1 - (step / num_of_steps),
        'cosine': lambda step: np.cos(0.5 * np.pi * step / (num_of_steps - 1)),
        'constant': lambda step: 1.0
    }
    
    scheduler = lr_schedulers.get(scheduler_type, lr_schedulers['constant'])
    
    opt = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    
    with tqdm(range(num_of_steps), desc="Training") as t:
        for step in t:
            step_lr = initial_lr * scheduler(step)
            for group in opt.param_groups:
                group['lr'] = step_lr
            
            opt.zero_grad(set_to_none=True)
            batch = model.generate_data(num_batch)
            out = model(batch)
            error = model.importance * (batch.abs() - out)**2
            loss = torch.mean(error)
            loss.backward()
            opt.step()
            
            wandb.log({
                'loss': loss.item(),
                'lr': step_lr
            })
            
            t.set_postfix(
                loss=loss.item(),
                lr=step_lr,
            )
    
    wandb.finish()
    
def plot_intro_diagram(model, save_path='intro_diagram.pdf'):
    WA = model.W.detach()
    print(f"Shape of WA: {WA.shape}")
    num_instances, num_features, hidden_dim = WA.shape

    importance_colors = sns.color_palette("viridis", n_colors=num_features)

    fig, axs = plt.subplots(1, num_instances, figsize=(2*num_instances, 2), dpi=200)
    if num_instances == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        print(f"Plotting for instance {i}")
        
        for j in range(num_features):
            ax.scatter(WA[i, j, 0], WA[i, j, 1], color=importance_colors[j], s=50)
            ax.plot([0, WA[i, j, 0]], [0, WA[i, j, 1]], color=importance_colors[j], alpha=0.5)
        
        ax.set_aspect('equal')
        ax.set_facecolor('#FCFBF8')
        z = max(1.5, np.abs(WA[i]).max() * 1.1)
        ax.set_xlim((-z, z))
        ax.set_ylim((-z, z))

        ax.tick_params(left=True, right=False, labelleft=False, labelbottom=False, bottom=True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_position('center')
        
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    
    print(f"Saving figure to {save_path}")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure saved successfully to {save_path}")

def main():
    model = ToyModel(num_features=5, num_hidden=2, num_instances=10)
    optimize(model, 'config.yaml')
    plot_intro_diagram(model)

if __name__ == "__main__":
    main()
