#!/usr/bin/env python3
"""
Large Language Model Training Examples
=====================================

This script demonstrates how to:
1. Build a simple transformer-based language model from scratch
2. Train it on text data
3. Generate text with the trained model
4. Fine-tune a pre-trained model

Author: AI Tutorial by AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import os
import json
from typing import Optional, List, Dict, Any

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# 1. TRANSFORMER COMPONENTS
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        return torch.matmul(attention_weights, V)


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward network"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward with residual connection
        fed_forward = self.ffn(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create a causal mask to prevent attention to future positions"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # True for allowed positions, False for masked


# ============================================================================
# 2. SIMPLE LANGUAGE MODEL
# ============================================================================

class SimpleLanguageModel(nn.Module):
    """A simple transformer-based language model"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 1024, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.ln_final = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len).to(input_ids.device)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.output_projection(x)
        
        return logits


# ============================================================================
# 3. TOKENIZER (SIMPLE WORD-LEVEL)
# ============================================================================

class SimpleTokenizer:
    """Simple word-level tokenizer for demonstration purposes"""
    
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        self.bos_token = '<BOS>'
        
        # Add special tokens
        self._add_word(self.pad_token)
        self._add_word(self.unk_token)
        self._add_word(self.eos_token)
        self._add_word(self.bos_token)
        
        self.pad_token_id = self.word_to_id[self.pad_token]
        self.unk_token_id = self.word_to_id[self.unk_token]
        self.eos_token_id = self.word_to_id[self.eos_token]
        self.bos_token_id = self.word_to_id[self.bos_token]
    
    def _add_word(self, word: str) -> int:
        if word not in self.word_to_id:
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1
        return self.word_to_id[word]
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from list of texts"""
        word_counts = {}
        
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add words that meet minimum frequency threshold
        for word, count in word_counts.items():
            if count >= min_freq:
                self._add_word(word)
        
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert text to list of token IDs"""
        words = text.lower().split()
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for word in words:
            token_id = self.word_to_id.get(word, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token IDs back to text"""
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            if skip_special_tokens and word in [self.pad_token, self.unk_token, self.eos_token, self.bos_token]:
                continue
            words.append(word)
        return ' '.join(words)


# ============================================================================
# 4. DATASET CLASS
# ============================================================================

class TextDataset(Dataset):
    """Dataset for language modeling"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            token_ids = tokenizer.encode(text)
            
            # Split long texts into chunks
            for i in range(0, len(token_ids) - max_length + 1, max_length // 2):
                chunk = token_ids[i:i + max_length]
                if len(chunk) == max_length:
                    self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================

def train_language_model(model: nn.Module, train_loader: DataLoader, 
                        val_loader: DataLoader, num_epochs: int = 10, 
                        learning_rate: float = 1e-4) -> nn.Module:
    """Train a language model"""
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch.to(device)
            
            # For language modeling, targets are input shifted by one position
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            # Forward pass
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch.to(device)
                targets = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        scheduler.step()
    
    return model


def calculate_perplexity(model: nn.Module, data_loader: DataLoader) -> float:
    """Calculate perplexity of the model on given data"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch.to(device)
            targets = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


# ============================================================================
# 6. TEXT GENERATION
# ============================================================================

def generate_text(model: nn.Module, tokenizer: SimpleTokenizer, prompt: str, 
                 max_length: int = 100, temperature: float = 1.0, 
                 top_k: int = 50) -> str:
    """Generate text using the trained model"""
    model.eval()
    model.to(device)
    
    # Tokenize the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), 
                           dtype=torch.long).unsqueeze(0).to(device)
    
    generated_tokens = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            logits = model(generated_tokens)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probabilities, 1)
            
            # Stop if we generate an end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append to generated sequence
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=1)
    
    # Decode and return generated text
    generated_text = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
    return generated_text


# ============================================================================
# 7. FINE-TUNING WITH TRANSFORMERS LIBRARY
# ============================================================================

def fine_tune_with_transformers():
    """Example of fine-tuning using the Transformers library"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from transformers import TextDataset, DataCollatorForLanguageModeling
        
        print("Fine-tuning with Transformers library...")
        
        # Load pre-trained model and tokenizer
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create a simple dataset (you would use your own data here)
        sample_texts = [
            "The future of artificial intelligence is bright and full of possibilities.",
            "Machine learning models can solve complex problems with the right data.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning has revolutionized the field of artificial intelligence.",
        ]
        
        # Save sample text to file for TextDataset
        with open('sample_data.txt', 'w') as f:
            for text in sample_texts:
                f.write(text + '\n')
        
        # Create dataset
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path='sample_data.txt',
            block_size=64
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./fine-tuned-gpt2',
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            save_steps=50,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_steps=10,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Start fine-tuning
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        tokenizer.save_pretrained('./fine-tuned-gpt2')
        
        print("Fine-tuning completed!")
        
        # Test generation
        print("\nGenerating text with fine-tuned model...")
        inputs = tokenizer.encode("The future of AI", return_tensors='pt')
        outputs = model.generate(inputs, max_length=50, temperature=0.8, do_sample=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        # Clean up
        os.remove('sample_data.txt')
        
    except ImportError:
        print("Transformers library not available. Install with: pip install transformers")
    except Exception as e:
        print(f"Error in fine-tuning: {e}")


# ============================================================================
# 8. MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main function demonstrating LLM training"""
    print("🤖 Large Language Model Training Examples")
    print("=" * 50)
    
    # Create sample text data
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "neural networks are inspired by biological neural networks",
        "deep learning uses multiple layers to learn representations",
        "transformers use attention mechanisms for better performance",
        "language models predict the next word in a sequence",
        "artificial intelligence will transform many industries",
        "data science combines statistics programming and domain knowledge",
        "python is a popular programming language for machine learning",
        "the future of technology depends on continued innovation"
    ] * 20  # Repeat to have more training data
    
    print(f"Created {len(sample_texts)} training examples")
    
    # Build tokenizer
    print("\n1. Building tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(sample_texts, min_freq=1)
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_texts = sample_texts[:int(0.8 * len(sample_texts))]
    val_texts = sample_texts[int(0.8 * len(sample_texts)):]
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=32)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=32)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    print("\n3. Creating language model...")
    model = SimpleLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=128
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Train model
    print("\n4. Training model...")
    model = train_language_model(model, train_loader, val_loader, num_epochs=5)
    
    # Calculate perplexity
    print("\n5. Evaluating model...")
    train_perplexity = calculate_perplexity(model, train_loader)
    val_perplexity = calculate_perplexity(model, val_loader)
    
    print(f"Train Perplexity: {train_perplexity:.2f}")
    print(f"Validation Perplexity: {val_perplexity:.2f}")
    
    # Generate text
    print("\n6. Generating text...")
    prompts = [
        "machine learning",
        "the future of",
        "artificial intelligence",
        "neural networks"
    ]
    
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=20, temperature=0.8)
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    # Save model
    print("\n7. Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'word_to_id': tokenizer.word_to_id,
        'id_to_word': tokenizer.id_to_word
    }, 'simple_language_model.pth')
    print("Model saved as 'simple_language_model.pth'")
    
    # Demonstrate fine-tuning with transformers
    print("\n8. Fine-tuning example with Transformers library...")
    fine_tune_with_transformers()
    
    print("\n🎉 LLM training examples completed!")
    print("\nNext steps:")
    print("- Try training on larger datasets")
    print("- Experiment with different model architectures")
    print("- Use pre-trained models for better results")
    print("- Implement more advanced techniques like PEFT")


if __name__ == "__main__":
    main()