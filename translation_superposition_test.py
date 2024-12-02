import torch
from torch import nn
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load as load_metric
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HuggingFaceTranslationDataset(Dataset):
    def __init__(self, split="train", max_length=128, max_samples=None):
        self.dataset = load_dataset("iwslt2017", "iwslt2017-en-fr")[split]
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
        self.tokenizer = MarianTokenizer.from_pretrained(base_model_name)
        self.hidden_size = hidden_size
        
        encoder_dim = self.base_model.config.d_model
        self.encoder_bottleneck = nn.Linear(encoder_dim, hidden_size)
        self.decoder_expansion = nn.Linear(hidden_size, encoder_dim)
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
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
            
            embedding_weights = self.base_model.model.shared.weight.detach().cpu().numpy()
            N = 200  
            selected_embeddings = embedding_weights[:N]
            tokens = [self.tokenizer.convert_ids_to_tokens(idx) for idx in range(N)]
            
            pca = PCA(n_components=3)
            reduced_embeddings = pca.fit_transform(selected_embeddings)
            
            writer.add_embedding(
                selected_embeddings,
                metadata=tokens,
                global_step=step,
                tag='Word Embeddings'
            )
            
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], alpha=0.7)
            for i, token in enumerate(tokens):
                if i % 10 == 0:
                    ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2], token)
            ax.set_title('Word Embeddings Visualized with PCA (3D)')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            plt.savefig(f'embeddings_step_{step}_3d.png')
            plt.close()

def evaluate_bleu(model, dataloader, num_samples=5):
    model.eval()
    bleu_metric = load_metric('sacrebleu')
    translations = []
    references = []
    sources = []
    device = model.device
    sample_outputs = [] 
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=50
            )

            decoded_preds = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_sources = model.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)

            decoded_labels = [[label] for label in decoded_labels]

            translations.extend(decoded_preds)
            references.extend(decoded_labels)
            sources.extend(decoded_sources)

            if len(sample_outputs) < num_samples:
                for src, pred, ref in zip(decoded_sources, decoded_preds, decoded_labels):
                    sample_outputs.append((src, pred, ref[0]))
                    if len(sample_outputs) >= num_samples:
                        break

    bleu = bleu_metric.compute(predictions=translations, references=references)
    print(f"BLEU score: {bleu['score']:.2f}")

    return bleu['score'], sample_outputs

def train_model(model, train_loader, val_loader, config):
    writer = SummaryWriter('runs/translation_superposition')
    optimizer = torch.optim.AdamW([
        {'params': model.encoder_bottleneck.parameters()},
        {'params': model.decoder_expansion.parameters()}
    ], lr=config['learning_rate'])
    
    device = model.device
    model = model.to(device)
    
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}") as t:
            for i, batch in enumerate(t):
                optimizer.zero_grad()
                
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss else loss.item()
                t.set_postfix(loss=f'{loss.item():.4f}', avg_loss=f'{running_loss:.4f}')
                
                global_step = epoch * len(train_loader) + i
                if i % config['viz_interval'] == 0:
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    model.visualize_bottleneck(writer, global_step)
        
        bleu_score, sample_outputs = evaluate_bleu(model, val_loader)
        writer.add_scalar('BLEU/validation', bleu_score, epoch)
        
        print(f"\nSample translations after epoch {epoch+1}:\n")
        for idx, (src, pred, ref) in enumerate(sample_outputs):
            print(f"Example {idx+1}:")
            print(f"Source:      {src}")
            print(f"Prediction:  {pred}")
            print(f"Reference:   {ref}\n")
    
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
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = SuperpositionTranslator(config['base_model'], config['hidden_size'])
    train_model(model, train_loader, val_loader, config)

if __name__ == "__main__":
    main()
