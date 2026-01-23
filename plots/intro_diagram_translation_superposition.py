import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import traceback

class HuggingFaceTranslationDataset(Dataset):
    def __init__(self, split="train", max_length=128, max_samples=None):
        print("Initializing dataset...")
        self.dataset = load_dataset("iwslt2017", "iwslt2017-en-fr", trust_remote_code=True)[split]
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        self.max_length = max_length
        print(f"Dataset initialized with {len(self.dataset)} samples")
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['translation']['en']
        tgt_text = item['translation']['fr']
        
        src_encoding = self.tokenizer(src_text, 
                                    max_length=self.max_length, 
                                    padding='max_length', 
                                    truncation=True, 
                                    return_tensors='pt')
        
        tgt_encoding = self.tokenizer(tgt_text,
                                    max_length=self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt')
        
        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': tgt_encoding['input_ids'].squeeze()
        }

class SuperpositionTranslator(nn.Module):
    def __init__(self, base_model_name, hidden_size=512):
        super().__init__()
        print(f"Initializing SuperpositionTranslator with hidden_size={hidden_size}")
        self.base_model = MarianMTModel.from_pretrained(base_model_name)
        self.hidden_size = hidden_size
        
        encoder_dim = self.base_model.config.d_model
        self.encoder_bottleneck = nn.Linear(encoder_dim, hidden_size)
        self.decoder_expansion = nn.Linear(hidden_size, encoder_dim)
        
        torch.nn.init.uniform_(self.encoder_bottleneck.weight, -0.5, 0.5)
        torch.nn.init.uniform_(self.decoder_expansion.weight, -0.5, 0.5)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        print(f"Model initialized with encoder_dim={encoder_dim}, hidden_size={hidden_size}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        encoder_outputs = self.base_model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = self.encoder_bottleneck(encoder_outputs[0])
        expanded_states = self.decoder_expansion(hidden_states)
        
        outputs = self.base_model(
            encoder_outputs=(expanded_states,),
            labels=labels,
            attention_mask=attention_mask
        )
        
        return outputs

class SuperpositionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.writer = SummaryWriter('runs/translation_experiment')
        self.device = next(model.parameters()).device
        print(f"Visualizer initialized. Model is on device: {self.device}")

    def plot_weight_vectors(self, step):
        try:
            with torch.no_grad():
                weights = self.model.encoder_bottleneck.weight.detach().cpu().numpy()
                weights = weights[:4, :2] 
                
                plt.clf()
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                
                colors = ['red', 'blue', 'green', 'purple']
                for i, (vec, color) in enumerate(zip(weights, colors)):
                    plt.quiver(0, 0, vec[0], vec[1], 
                             angles='xy', scale_units='xy', scale=1,
                             color=color, label=f'Vector {i+1}')
                
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.title(f'Weight Vectors at Step {step}')
                
                plt.savefig(f'weight_plot_{step}.png')
                self.writer.add_figure('weights', fig, step)
                plt.close()
                
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            traceback.print_exc()
            plt.close('all')

    def close(self):
        self.writer.close()

def train_model(model, train_loader, tokenizer, config):
    print("Starting training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.base_model = model.base_model.to(device)
    print(f"Model moved to device: {device}")
    
    visualizer = SuperpositionVisualizer(model, tokenizer)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder_bottleneck.parameters()},
        {'params': model.decoder_expansion.parameters()}
    ], lr=config['learning_rate'])
    
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as t:
            for i, batch in enumerate(t):
                optimizer.zero_grad()
                
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                running_loss = 0.9 * running_loss + 0.1 * loss.item()
                t.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{running_loss:.4f}')
                
                if i % config['viz_interval'] == 0:
                    step = epoch * len(train_loader) + i
                    visualizer.writer.add_scalar('Loss/train', loss.item(), step)
                    visualizer.plot_weight_vectors(step)
    
    visualizer.close()
    print("Training completed!")

def main():
    config = {
        'base_model': 'Helsinki-NLP/opus-mt-en-fr',
        'hidden_size': 16,      
        'batch_size': 8,        
        'num_epochs': 3,
        'learning_rate': 0.01,  
        'max_samples': 500,     
        'viz_interval': 10      
    }
    
    print("Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(config['base_model'])
    
    print("Creating dataset...")
    dataset = HuggingFaceTranslationDataset(max_samples=config['max_samples'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    print("Initializing model...")
    model = SuperpositionTranslator(config['base_model'], config['hidden_size'])
    
    print("Starting training process...")
    train_model(model, dataloader, tokenizer, config)

if __name__ == "__main__":
    main()