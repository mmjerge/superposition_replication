import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

class HuggingFaceTranslationDataset(Dataset):
    def __init__(self, split="train", max_length=128, max_samples=None):
        self.dataset = load_dataset("iwslt2017", "iwslt2017-en-fr", trust_remote_code=True)[split]
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
        self.max_length = max_length
        
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
        self.base_model = MarianMTModel.from_pretrained(base_model_name)
        self.hidden_size = hidden_size
        
        encoder_dim = self.base_model.config.d_model
        self.encoder_bottleneck = nn.Linear(encoder_dim, hidden_size)
        self.decoder_expansion = nn.Linear(hidden_size, encoder_dim)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
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
    
    def visualize_bottleneck(self, writer: SummaryWriter, step: int):
        with torch.no_grad():
            weights = self.encoder_bottleneck.weight.detach().cpu().numpy()
            writer.add_histogram('bottleneck/weight_distribution', weights, step)
            
            corr_matrix = np.corrcoef(weights)
            writer.add_image('bottleneck/correlation_matrix', 
                           corr_matrix[None, :, :], 
                           step, 
                           dataformats='CHW')

def train_model(model, train_loader, config):
    writer = SummaryWriter('runs/translation_superposition')
    optimizer = torch.optim.AdamW([
        {'params': model.encoder_bottleneck.parameters()},
        {'params': model.decoder_expansion.parameters()}
    ], lr=config['learning_rate'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
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
                    writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
                    model.visualize_bottleneck(writer, epoch * len(train_loader) + i)
    
    writer.close()

def main():
    config = {
        'base_model': 'Helsinki-NLP/opus-mt-en-fr',
        'hidden_size': 256,
        'batch_size': 16,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'max_samples': 10000,
        'viz_interval': 100
    }
    
    dataset = HuggingFaceTranslationDataset(max_samples=config['max_samples'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    model = SuperpositionTranslator(config['base_model'], config['hidden_size'])
    train_model(model, dataloader, config)

if __name__ == "__main__":
    main()