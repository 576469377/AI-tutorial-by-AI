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
    
    # Demonstrate multimodal capabilities
    print("\n9. Multimodal AI examples...")
    multimodal_examples()
    
    print("\n🎉 LLM training examples completed!")
    print("\nNext steps:")
    print("- Try training on larger datasets")
    print("- Experiment with different model architectures")
    print("- Use pre-trained models for better results")
    print("- Implement more advanced techniques like PEFT")
    print("- Explore multimodal applications like image captioning")


def multimodal_examples():
    """Demonstrate multimodal AI capabilities"""
    print("🎨 Multimodal AI Examples")
    print("=" * 40)
    
    # Example 1: CLIP-based image-text similarity
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        import requests
        import numpy as np
        
        print("\n1. CLIP Image-Text Similarity Demo")
        print("-" * 35)
        
        # Load CLIP model and processor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Sample image URL (a cat)
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg"
        
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            
            # Text descriptions to compare
            text_descriptions = [
                "a cat sitting on a surface",
                "a dog playing in the park", 
                "a beautiful landscape",
                "a fluffy orange cat",
                "a bird flying in the sky"
            ]
            
            # Process inputs
            inputs = processor(
                text=text_descriptions, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            
            # Get similarities
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            print("Image-Text Similarity Scores:")
            for i, desc in enumerate(text_descriptions):
                score = probs[0][i].item()
                print(f"  '{desc}': {score:.3f}")
                
        except Exception as e:
            print(f"Could not load image: {e}")
            print("Simulating image-text similarity scores...")
            for i, desc in enumerate(text_descriptions):
                # Simulate higher scores for cat-related descriptions
                if "cat" in desc.lower():
                    score = np.random.uniform(0.6, 0.9)
                else:
                    score = np.random.uniform(0.1, 0.4)
                print(f"  '{desc}': {score:.3f}")
                    
    except ImportError:
        print("CLIP model not available. Install transformers library.")
    except Exception as e:
        print(f"CLIP demo error: {e}")
    
    # Example 2: Simple Vision-Language Model Architecture
    print("\n2. Vision-Language Model Architecture")
    print("-" * 40)
    
    class SimpleVisionLanguageModel(nn.Module):
        """Simplified vision-language model for demonstration"""
        
        def __init__(self, vocab_size, d_model=512):
            super().__init__()
            
            # Vision encoder (simplified)
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, d_model)
            )
            
            # Text encoder
            self.text_encoder = nn.Sequential(
                nn.Embedding(vocab_size, d_model),
                nn.LSTM(d_model, d_model, batch_first=True),
            )
            
            # Fusion layer
            self.fusion = nn.MultiheadAttention(d_model, num_heads=8)
            
            # Output projections
            self.image_projection = nn.Linear(d_model, d_model)
            self.text_projection = nn.Linear(d_model, d_model)
            
        def forward(self, image, text):
            # Encode image and text
            image_features = self.vision_encoder(image)  # [batch, d_model]
            text_features, _ = self.text_encoder(text)   # [batch, seq_len, d_model]
            
            # Cross-modal attention
            image_features = image_features.unsqueeze(1)  # [batch, 1, d_model]
            fused_features, _ = self.fusion(
                image_features, text_features, text_features
            )
            
            # Project to common space
            image_proj = self.image_projection(fused_features.squeeze(1))
            text_proj = self.text_projection(text_features.mean(dim=1))
            
            return image_proj, text_proj
    
    # Create and demonstrate the model
    vocab_size = 10000
    vl_model = SimpleVisionLanguageModel(vocab_size)
    
    # Create dummy inputs
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_text = torch.randint(0, vocab_size, (batch_size, 20))
    
    # Forward pass
    image_proj, text_proj = vl_model(dummy_image, dummy_text)
    
    print(f"Vision-Language Model created successfully!")
    print(f"  Image projection shape: {image_proj.shape}")
    print(f"  Text projection shape: {text_proj.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in vl_model.parameters()):,}")
    
    # Example 3: Image Captioning Training Setup
    print("\n3. Image Captioning Training Setup")
    print("-" * 38)
    
    class ImageCaptioningModel(nn.Module):
        """Simple image captioning model"""
        
        def __init__(self, vocab_size, d_model=512, max_seq_len=50):
            super().__init__()
            self.d_model = d_model
            self.max_seq_len = max_seq_len
            
            # Vision encoder (using ResNet-like architecture)
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, d_model)
            )
            
            # Caption decoder (using transformer decoder)
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model, 
                nhead=8, 
                dim_feedforward=2048,
                dropout=0.1
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            
            # Output projection
            self.output_projection = nn.Linear(d_model, vocab_size)
        
        def forward(self, image, caption_tokens=None):
            # Encode image
            image_features = self.vision_encoder(image)  # [batch, d_model]
            image_features = image_features.unsqueeze(1)  # [batch, 1, d_model]
            
            if caption_tokens is not None:
                # Training mode: teacher forcing
                seq_len = caption_tokens.size(1)
                caption_embeds = self.embedding(caption_tokens)
                caption_embeds += self.positional_encoding[:seq_len].unsqueeze(0)
                
                # Create causal mask
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                
                # Decode with cross-attention to image
                output = self.transformer_decoder(
                    caption_embeds.transpose(0, 1),  # [seq_len, batch, d_model]
                    image_features.transpose(0, 1),  # [1, batch, d_model] 
                    tgt_mask=causal_mask
                )
                
                # Project to vocabulary
                logits = self.output_projection(output.transpose(0, 1))
                return logits
            else:
                # Inference mode: autoregressive generation
                # This would be implemented for actual text generation
                return image_features
    
    # Create captioning model
    captioning_model = ImageCaptioningModel(vocab_size=5000)
    total_params = sum(p.numel() for p in captioning_model.parameters())
    
    print(f"Image Captioning Model created!")
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_captions = torch.randint(0, 5000, (2, 20))
    
    logits = captioning_model(dummy_image, dummy_captions)
    print(f"  Output logits shape: {logits.shape}")
    
    # Example 4: Visual Question Answering
    print("\n4. Visual Question Answering Setup")
    print("-" * 36)
    
    class VQAModel(nn.Module):
        """Visual Question Answering model"""
        
        def __init__(self, vocab_size, num_answers=1000, d_model=512):
            super().__init__()
            
            # Vision encoder
            self.vision_encoder = nn.Sequential(
                nn.Conv2d(3, 512, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((14, 14)),
                nn.Flatten(),
                nn.Linear(512 * 14 * 14, d_model)
            )
            
            # Question encoder
            self.question_encoder = nn.Sequential(
                nn.Embedding(vocab_size, d_model),
                nn.LSTM(d_model, d_model // 2, bidirectional=True, batch_first=True)
            )
            
            # Attention mechanism
            self.attention = nn.MultiheadAttention(d_model, num_heads=8)
            
            # Answer classifier
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, num_answers)
            )
        
        def forward(self, image, question):
            # Encode image and question
            image_features = self.vision_encoder(image)
            question_features, _ = self.question_encoder(question)
            
            # Attention between question and image
            question_attended, _ = self.attention(
                question_features.mean(dim=1, keepdim=True),
                image_features.unsqueeze(1),
                image_features.unsqueeze(1)
            )
            
            # Classify answer
            answer_logits = self.classifier(question_attended.squeeze(1))
            return answer_logits
    
    # Create VQA model
    vqa_model = VQAModel(vocab_size=10000, num_answers=3000)
    total_params = sum(p.numel() for p in vqa_model.parameters())
    
    print(f"VQA Model created!")
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward pass
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_question = torch.randint(0, 10000, (2, 15))
    
    answer_logits = vqa_model(dummy_image, dummy_question)
    print(f"  Answer logits shape: {answer_logits.shape}")
    
    # Example 5: Multimodal Evaluation Metrics
    print("\n5. Multimodal Evaluation Metrics")
    print("-" * 35)
    
    def calculate_bleu_score(reference, candidate):
        """Simplified BLEU score calculation"""
        from collections import Counter
        
        # Tokenize (simple whitespace tokenization)
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        # Calculate precision for 1-grams
        ref_counts = Counter(ref_tokens)
        cand_counts = Counter(cand_tokens)
        
        overlap = sum(min(ref_counts[token], cand_counts[token]) 
                     for token in cand_counts)
        precision = overlap / len(cand_tokens) if len(cand_tokens) > 0 else 0
        
        # Simplified BLEU (just 1-gram precision with brevity penalty)
        brevity_penalty = min(1.0, len(cand_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0
        bleu = precision * brevity_penalty
        
        return bleu
    
    # Example captions for evaluation
    reference_caption = "a brown dog sitting on green grass"
    candidate_captions = [
        "a dog sitting on grass",
        "a brown animal on the ground", 
        "a cat playing with a ball",
        "brown dog on green grass"
    ]
    
    print("BLEU Score Evaluation:")
    for i, candidate in enumerate(candidate_captions):
        bleu = calculate_bleu_score(reference_caption, candidate)
        print(f"  Candidate {i+1}: BLEU = {bleu:.3f}")
        print(f"    Reference: '{reference_caption}'")
        print(f"    Candidate: '{candidate}'")
        print()
    
    print("✅ Multimodal examples completed!")
    print("\n🔗 Key Multimodal Applications:")
    print("  - Image Captioning: Describing images with natural language")
    print("  - Visual Question Answering: Answering questions about images")
    print("  - Image-Text Retrieval: Finding relevant images for text queries")
    print("  - Multimodal Chatbots: Conversational AI with vision capabilities")
    print("  - Content Generation: Creating images from text descriptions")
    
    # Section 10: RLHF and Advanced Alignment Examples
    print("\n10. RLHF and Model Alignment Examples...")
    rlhf_examples()


# ============================================================================
# 10. RLHF AND MODEL ALIGNMENT EXAMPLES
# ============================================================================

class SimpleRewardModel(nn.Module):
    """Simple reward model for RLHF demonstrations"""
    
    def __init__(self, vocab_size: int, d_model: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=2
        )
        self.reward_head = nn.Linear(d_model, 1)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        # Use mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
            x = (x * mask_expanded).sum(1) / mask_expanded.sum(1)
        else:
            x = x.mean(1)
        reward = self.reward_head(x)
        return reward.squeeze(-1)


def create_preference_dataset():
    """Create synthetic preference data for RLHF training"""
    
    # Synthetic preference data: prompt -> (chosen_response, rejected_response)
    preference_data = [
        {
            "prompt": "Explain machine learning",
            "chosen": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "rejected": "Machine learning is when computers become smart and can think like humans."
        },
        {
            "prompt": "How to stay healthy?",
            "chosen": "Maintain a balanced diet, exercise regularly, get adequate sleep, and have regular health checkups.",
            "rejected": "Just eat whatever you want and don't worry about it."
        },
        {
            "prompt": "What is Python?",
            "chosen": "Python is a high-level programming language known for its simplicity and versatility, widely used in data science and AI.",
            "rejected": "Python is a snake that programmers worship for some reason."
        },
        {
            "prompt": "How to learn programming?",
            "chosen": "Start with fundamentals, practice regularly, build projects, and learn from community resources and documentation.",
            "rejected": "Just copy code from the internet until something works."
        }
    ]
    
    return preference_data


def train_reward_model_demo(tokenizer):
    """Demonstrate reward model training"""
    print("\n🎯 Reward Model Training Demo")
    print("-" * 40)
    
    # Create synthetic preference dataset
    preference_data = create_preference_dataset()
    print(f"Created {len(preference_data)} preference pairs")
    
    # Initialize reward model
    reward_model = SimpleRewardModel(vocab_size=tokenizer.vocab_size)
    optimizer = optim.Adam(reward_model.parameters(), lr=1e-4)
    
    print("\nTraining reward model on preferences...")
    
    # Training loop
    for epoch in range(3):
        total_loss = 0
        correct_preferences = 0
        
        for example in preference_data:
            # Tokenize inputs
            prompt = example['prompt']
            chosen_text = prompt + " " + example['chosen']
            rejected_text = prompt + " " + example['rejected']
            
            chosen_tokens = tokenizer.encode(chosen_text)[:32]  # Truncate
            rejected_tokens = tokenizer.encode(rejected_text)[:32]
            
            # Pad sequences
            max_len = max(len(chosen_tokens), len(rejected_tokens))
            chosen_tokens += [0] * (max_len - len(chosen_tokens))
            rejected_tokens += [0] * (max_len - len(rejected_tokens))
            
            chosen_input = torch.tensor([chosen_tokens])
            rejected_input = torch.tensor([rejected_tokens])
            
            # Get reward scores
            chosen_reward = reward_model(chosen_input)
            rejected_reward = reward_model(rejected_input)
            
            # Bradley-Terry loss: P(chosen > rejected)
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check if model prefers chosen over rejected
            if chosen_reward.item() > rejected_reward.item():
                correct_preferences += 1
        
        avg_loss = total_loss / len(preference_data)
        accuracy = correct_preferences / len(preference_data)
        
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Preference Accuracy = {accuracy:.2%}")
    
    print("✅ Reward model training completed!")
    
    # Test reward model
    print("\n🧪 Testing Reward Model:")
    test_responses = [
        "This is a helpful and informative response.",
        "This is a low-quality response.",
        "I provide accurate and detailed information.",
        "I don't know anything about this topic."
    ]
    
    for response in test_responses:
        tokens = tokenizer.encode(response)[:32]
        tokens += [0] * (32 - len(tokens))  # Pad
        input_tensor = torch.tensor([tokens])
        
        with torch.no_grad():
            reward = reward_model(input_tensor).item()
        
        print(f"  Response: '{response[:50]}...'")
        print(f"  Reward Score: {reward:.3f}")
        print()
    
    return reward_model


def ppo_training_demo(model, reward_model, tokenizer):
    """Demonstrate simplified PPO training for RLHF"""
    print("\n🚀 PPO Training Demo (Simplified)")
    print("-" * 40)
    
    prompts = [
        "Explain artificial intelligence",
        "How to learn programming", 
        "What is machine learning",
        "Benefits of exercise"
    ]
    
    print("Simulating PPO training steps...")
    
    # Simplified PPO simulation
    for step in range(3):
        print(f"\nPPO Step {step + 1}:")
        
        total_reward = 0
        for prompt in prompts:
            # Simulate text generation (normally would use model.generate())
            generated_responses = [
                f"{prompt}: This is a helpful and detailed explanation.",
                f"{prompt}: This is a brief but accurate answer.",
                f"{prompt}: This provides comprehensive information."
            ]
            
            # Calculate rewards for each response
            for response in generated_responses:
                tokens = tokenizer.encode(response)[:32]
                tokens += [0] * (32 - len(tokens))
                input_tensor = torch.tensor([tokens])
                
                with torch.no_grad():
                    reward = reward_model(input_tensor).item()
                
                total_reward += reward
        
        avg_reward = total_reward / (len(prompts) * len(generated_responses))
        print(f"  Average Reward: {avg_reward:.3f}")
        
        # In real PPO, we would:
        # 1. Compute log probabilities and importance ratios
        # 2. Apply clipped objective function
        # 3. Update policy with gradient descent
        # 4. Maintain KL divergence constraints
        
        print(f"  📈 Policy improvement step completed")
    
    print("\n✅ PPO training simulation completed!")
    print("\n🔍 Key PPO Components:")
    print("  - Importance Sampling: ratio = π_new(a|s) / π_old(a|s)")
    print("  - Clipped Objective: min(ratio * advantage, clip(ratio) * advantage)")
    print("  - KL Penalty: β * KL(π_new || π_old)")
    print("  - Value Function: V(s) for advantage estimation")


def dpo_training_demo(tokenizer):
    """Demonstrate Direct Preference Optimization (DPO)"""
    print("\n🎯 Direct Preference Optimization (DPO) Demo")
    print("-" * 40)
    
    print("DPO directly optimizes preferences without reward modeling!")
    
    # Create preference dataset
    preference_data = create_preference_dataset()
    
    # Simulate DPO loss calculation
    print("\n📊 DPO Loss Calculation:")
    
    for i, example in enumerate(preference_data[:2]):  # Show first 2 examples
        print(f"\nExample {i + 1}:")
        print(f"  Prompt: {example['prompt']}")
        print(f"  Chosen: {example['chosen']}")
        print(f"  Rejected: {example['rejected']}")
        
        # Simulate log probabilities (in real implementation, these come from model)
        chosen_logp_policy = -2.1  # Simulated
        rejected_logp_policy = -3.2
        chosen_logp_ref = -2.5     # Reference model
        rejected_logp_ref = -3.0
        
        # DPO loss calculation
        beta = 0.1
        policy_diff = chosen_logp_policy - rejected_logp_policy
        ref_diff = chosen_logp_ref - rejected_logp_ref
        
        dpo_loss = -math.log(1 / (1 + math.exp(-beta * (policy_diff - ref_diff))))
        
        print(f"  Policy Diff: {policy_diff:.3f}")
        print(f"  Reference Diff: {ref_diff:.3f}")
        print(f"  DPO Loss: {dpo_loss:.3f}")
    
    print("\n✅ DPO advantages over RLHF:")
    print("  - No reward model needed")
    print("  - More stable training")
    print("  - Direct preference optimization")
    print("  - Simpler implementation")


def constitutional_ai_demo():
    """Demonstrate Constitutional AI principles"""
    print("\n📜 Constitutional AI Demo")
    print("-" * 40)
    
    # Define AI constitution
    constitution = [
        "Be helpful, harmless, and honest",
        "Respect human autonomy and dignity", 
        "Provide accurate information",
        "Admit uncertainty when appropriate",
        "Avoid generating harmful content"
    ]
    
    print("🏛️  AI Constitution:")
    for i, principle in enumerate(constitution, 1):
        print(f"  {i}. {principle}")
    
    # Example scenarios
    scenarios = [
        {
            "user_input": "How to hack into someone's computer?",
            "initial_response": "I can't provide instructions for unauthorized computer access as it's illegal and harmful.",
            "critique": "Good - follows principle of avoiding harmful content and respects legal boundaries.",
            "revision": "N/A - Initial response already follows constitutional principles."
        },
        {
            "user_input": "What's the capital of Mars?",
            "initial_response": "The capital of Mars is New Geneva.",
            "critique": "Problematic - provides false information. Should admit uncertainty about fictional concepts.",
            "revision": "Mars doesn't have a capital city as it's a planet without human settlements or governments."
        }
    ]
    
    print("\n🔍 Constitutional AI in Action:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}:")
        print(f"  User Input: {scenario['user_input']}")
        print(f"  Initial Response: {scenario['initial_response']}")
        print(f"  Constitutional Critique: {scenario['critique']}")
        print(f"  Revised Response: {scenario['revision']}")
    
    print("\n✅ Constitutional AI Process:")
    print("  1. Generate initial response")
    print("  2. Critique against constitution")
    print("  3. Revise if needed")
    print("  4. Repeat until constitutional compliance")


def chain_of_thought_demo():
    """Demonstrate Chain-of-Thought reasoning"""
    print("\n🧠 Chain-of-Thought (CoT) Reasoning Demo")
    print("-" * 40)
    
    problem = "If a train travels 240 miles in 4 hours, what is its average speed?"
    
    print(f"Problem: {problem}")
    
    # Standard response
    print("\n❌ Standard Response:")
    print("The speed is 60 mph.")
    
    # Chain-of-thought response
    print("\n✅ Chain-of-Thought Response:")
    print("Let me think step by step:")
    print("1. I need to find average speed")
    print("2. Speed = Distance / Time")
    print("3. Distance = 240 miles")
    print("4. Time = 4 hours")
    print("5. Speed = 240 miles ÷ 4 hours = 60 mph")
    print("Therefore, the average speed is 60 mph.")
    
    print("\n🎯 CoT Benefits:")
    print("  - Improved reasoning quality")
    print("  - Better problem decomposition")
    print("  - More transparent thinking")
    print("  - Reduced errors in complex problems")
    
    # Few-shot CoT example
    print("\n📚 Few-Shot CoT Example:")
    examples = [
        "Q: 15 + 27 = ?\nA: Let me add step by step: 15 + 27 = 15 + 20 + 7 = 35 + 7 = 42",
        "Q: 8 × 9 = ?\nA: Let me multiply: 8 × 9 = 8 × 10 - 8 × 1 = 80 - 8 = 72"
    ]
    
    for example in examples:
        print(f"  {example}")


def alignment_techniques_demo():
    """Demonstrate various alignment techniques"""
    print("\n🎯 Model Alignment Techniques Overview")
    print("-" * 40)
    
    techniques = {
        "RLHF": {
            "description": "Reinforcement Learning from Human Feedback",
            "components": ["SFT", "Reward Modeling", "PPO Training"],
            "benefits": "Aligns model behavior with human preferences",
            "challenges": "Complex pipeline, reward hacking risk"
        },
        "DPO": {
            "description": "Direct Preference Optimization", 
            "components": ["Preference Data", "Direct Training"],
            "benefits": "Simpler than RLHF, more stable training",
            "challenges": "Still requires high-quality preference data"
        },
        "Constitutional AI": {
            "description": "Training with explicit principles",
            "components": ["Constitution", "Critique", "Revision"],
            "benefits": "Transparent principles, self-improvement",
            "challenges": "Defining comprehensive constitution"
        },
        "RLAIF": {
            "description": "Reinforcement Learning from AI Feedback",
            "components": ["AI Labeler", "Preference Generation", "RL Training"],
            "benefits": "Scalable feedback, consistent preferences",
            "challenges": "Quality depends on AI labeler capability"
        }
    }
    
    for name, details in techniques.items():
        print(f"\n🔧 {name}: {details['description']}")
        print(f"  Components: {', '.join(details['components'])}")
        print(f"  Benefits: {details['benefits']}")
        print(f"  Challenges: {details['challenges']}")
    
    print("\n📈 Alignment Research Frontiers:")
    print("  - Scalable oversight methods")
    print("  - Interpretability and transparency")
    print("  - Robustness to distributional shift")
    print("  - Value learning and specification")
    print("  - Multi-agent alignment scenarios")


def rlhf_examples():
    """Main function for RLHF and alignment examples"""
    print("\n🤖 RLHF and Model Alignment Examples")
    print("=" * 50)
    
    # Create a simple tokenizer for demonstrations
    tokenizer = SimpleTokenizer()
    sample_texts = [
        "this is helpful information about machine learning",
        "here is a detailed explanation of the topic",
        "i provide accurate and useful responses",
        "this response is not very informative",
        "i don't know much about this subject"
    ]
    tokenizer.build_vocab(sample_texts, min_freq=1)
    
    # 1. Reward Model Training
    reward_model = train_reward_model_demo(tokenizer)
    
    # 2. PPO Training Demo
    ppo_training_demo(None, reward_model, tokenizer)
    
    # 3. DPO Demo
    dpo_training_demo(tokenizer)
    
    # 4. Constitutional AI Demo
    constitutional_ai_demo()
    
    # 5. Chain-of-Thought Demo
    chain_of_thought_demo()
    
    # 6. Alignment Techniques Overview
    alignment_techniques_demo()
    
    print("\n✅ RLHF and alignment examples completed!")
    print("\n🌟 Key Takeaways:")
    print("  - RLHF aligns models with human preferences through 3-stage process")
    print("  - DPO offers simpler alternative to RLHF without reward modeling")
    print("  - Constitutional AI provides transparent principle-based training")
    print("  - Chain-of-Thought improves reasoning capabilities")
    print("  - Model alignment is crucial for safe and beneficial AI systems")


if __name__ == "__main__":
    main()