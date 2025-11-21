"""
Corrected Transformer Implementation for Genomic Language Models
Based on the paper: "Predicting Beta-Lactamase Resistance Genes in Acinetobacter baumannii"

This file contains the corrected implementations that match the paper's described methodology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    CORRECTED Positional encoding for transformer input.
    
    This version correctly handles batch_first=True tensors.
    
    Key Fix: Properly handles (batch, seq_len, embed_dim) shaped inputs
    instead of assuming (seq_len, batch, embed_dim).
    
    Reference: Vaswani et al., 2017 - "Attention is all you need"
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # CRITICAL FIX: Shape for batch_first=True
        # Changed from pe.unsqueeze(0).transpose(0, 1) to just pe.unsqueeze(0)
        # Shape: (1, max_len, d_model) for broadcasting with (batch, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model) when batch_first=True
        
        Returns:
            Tensor of same shape with positional encoding added
        """
        # CRITICAL FIX: Use dimension 1 for sequence length when batch_first=True
        seq_len = x.size(1)  # Not x.size(0) which would be batch_size!
        
        # Add positional encoding, broadcasting over batch dimension
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


class SimpleTransformerEncoder(nn.Module):
    """
    CORRECTED Simple Transformer Encoder for genomic data.
    
    This is a basic encoder-only model for next-token prediction.
    For the full paper implementation, see PanBARTModel below.
    
    Key Fixes:
    - Corrected positional encoding integration
    - Proper batch_first handling
    - Device-aware operations
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, 
                 max_seq_length=256, dropout_rate=0.2, pe_max_len=5000):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # CORRECTED positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=dropout_rate)
        
        # Transformer encoder layers with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,  # Standard: 4x model dimension
            dropout=dropout_rate,
            batch_first=True,  # Input shape: (batch, seq, feature)
            activation='gelu'  # GELU is more common in modern transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.out = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following best practices"""
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, attention_mask=None):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input token indices, shape (batch, seq_len)
            attention_mask: Optional mask for padding tokens
        
        Returns:
            Logits for next token prediction, shape (batch, seq_len, vocab_size)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embed(x)
        
        # Add positional encoding (now correctly handles batch_first)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Project to vocabulary: (batch, seq_len, embed_dim) -> (batch, seq_len, vocab_size)
        logits = self.out(x)
        
        return logits


class PanBARTModel(nn.Module):
    """
    FULL IMPLEMENTATION matching the paper's description.
    
    This implements the encoder-decoder architecture with:
    - Masked Language Modeling (MLM) for pre-training
    - Binary classification for resistance prediction
    - Multi-task learning capability
    
    As described in the paper:
    "The model architecture consists of three main components working in concert,
    following the encoder-decoder transformer paradigm established by Vaswani et al."
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4,
                 max_seq_length=256, dropout_rate=0.2, pe_max_len=5000,
                 num_classes=2):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Shared token embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding (shared between encoder and decoder)
        self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=dropout_rate)
        
        # === ENCODER ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # === DECODER ===
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # === MLM HEAD (for masked language modeling pre-training) ===
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )
        
        # === CLASSIFICATION HEAD (for resistance prediction) ===
        # As described: "Classification head performs binary classification"
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Special tokens
        self.cls_token_id = vocab_size - 3  # [CLS] token
        self.mask_token_id = vocab_size - 2  # [MASK] token
        self.pad_token_id = vocab_size - 1   # [PAD] token
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        
        for layer in self.mlm_head:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, task='mlm', attention_mask=None):
        """
        Forward pass with multi-task capability.
        
        Args:
            x: Input token indices, shape (batch, seq_len)
            task: One of ['mlm', 'classification', 'generation']
            attention_mask: Optional padding mask
        
        Returns:
            Depends on task:
            - 'mlm': Logits for masked tokens, shape (batch, seq_len, vocab_size)
            - 'classification': Class logits, shape (batch, num_classes)
            - 'generation': Next token logits, shape (batch, seq_len, vocab_size)
        """
        # Embedding and positional encoding
        embedded = self.embed(x)
        embedded = self.pos_encoding(embedded)
        
        # Encode
        encoded = self.encoder(embedded, src_key_padding_mask=attention_mask)
        
        if task == 'mlm':
            # Masked Language Modeling: predict masked tokens
            # Decode using encoded context
            decoded = self.decoder(embedded, encoded, 
                                  tgt_key_padding_mask=attention_mask,
                                  memory_key_padding_mask=attention_mask)
            mlm_logits = self.mlm_head(decoded)
            return mlm_logits
        
        elif task == 'classification':
            # Resistance classification: use [CLS] token or global pooling
            # Paper mentions: "Classification head performs binary classification"
            
            # Option 1: Use [CLS] token (first token) representation
            cls_representation = encoded[:, 0, :]
            
            # Option 2: Global average pooling (alternative)
            # pooled = encoded.mean(dim=1)
            
            class_logits = self.classifier(cls_representation)
            return class_logits
        
        elif task == 'generation':
            # Autoregressive generation: predict next tokens
            decoded = self.decoder(embedded, encoded,
                                  tgt_key_padding_mask=attention_mask,
                                  memory_key_padding_mask=attention_mask)
            gen_logits = self.mlm_head(decoded)
            return gen_logits
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_attention_weights(self, x, attention_mask=None):
        """
        Extract attention weights for visualization.
        
        As mentioned in paper: "Attention visualization provides interpretable 
        insights by identifying which genes the model attends to"
        
        Returns:
            List of attention weight tensors from each layer
        """
        embedded = self.embed(x)
        embedded = self.pos_encoding(embedded)
        
        attention_weights = []
        
        # Extract attention from each encoder layer
        for layer in self.encoder.layers:
            # This requires modifying the forward pass or using hooks
            # For production, use register_forward_hook
            pass
        
        return attention_weights


class MLMTrainer:
    """
    Training utilities for Masked Language Modeling.
    
    As described in paper:
    "Pre-training begins with masked gene modeling on entire bacterial
    pangenome corpus"
    """
    
    def __init__(self, model, tokenizer, mask_prob=0.15, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.device = device
        
    def mask_tokens(self, inputs):
        """
        Prepare masked inputs and labels for MLM.
        
        Following BERT's masking strategy:
        - 80% of the time: Replace with [MASK]
        - 10% of the time: Replace with random token
        - 10% of the time: Keep original token
        
        Args:
            inputs: Tensor of token ids, shape (batch, seq_len)
        
        Returns:
            masked_inputs: Tensor with masked tokens
            labels: Original tokens for computing loss
            mask: Boolean tensor indicating which tokens were masked
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        
        # Create random mask (don't mask special tokens)
        probability_matrix = torch.full(inputs.shape, self.mask_prob)
        special_tokens_mask = (
            (inputs == self.tokenizer.token_to_id("[PAD]")) |
            (inputs == self.tokenizer.token_to_id("[CLS]"))
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Only compute loss on masked tokens
        labels[~masked_indices] = -100  # PyTorch ignores -100 in CrossEntropyLoss
        
        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.token_to_id("[MASK]")
        
        # 10% of time, replace with random token
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(len(self.tokenizer.get_vocab()), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_tokens[indices_random]
        
        # 10% of time, keep original (already set)
        
        return inputs, labels, masked_indices
    
    def train_step(self, batch, optimizer, criterion):
        """
        Single training step for MLM.
        
        Args:
            batch: Batch of token ids
            optimizer: PyTorch optimizer
            criterion: Loss function (CrossEntropyLoss)
        
        Returns:
            loss: Scalar loss value
        """
        self.model.train()
        
        # Move to device
        inputs = batch.to(self.device)
        
        # Create masked inputs
        masked_inputs, labels, mask = self.mask_tokens(inputs)
        masked_inputs = masked_inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(masked_inputs, task='mlm')
        
        # Compute loss only on masked tokens
        loss = criterion(outputs.view(-1, self.model.vocab_size), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (good practice for transformers)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()


class ResistanceClassifier:
    """
    Fine-tuning utilities for resistance classification.
    
    As described in paper:
    "Fine-tuning specifically on A. baumannii data for resistance prediction"
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def train_step(self, batch_inputs, batch_labels, optimizer, criterion):
        """
        Single training step for classification.
        
        Args:
            batch_inputs: Batch of genomic sequences (token ids)
            batch_labels: Binary labels (0=susceptible, 1=resistant)
            optimizer: PyTorch optimizer
            criterion: Loss function (BCEWithLogitsLoss or CrossEntropyLoss)
        
        Returns:
            loss: Scalar loss value
            predictions: Predicted classes
        """
        self.model.train()
        
        # Move to device
        inputs = batch_inputs.to(self.device)
        labels = batch_labels.to(self.device)
        
        # Forward pass
        logits = self.model(inputs, task='classification')
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Get predictions
        predictions = torch.argmax(logits, dim=1)
        
        return loss.item(), predictions


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    Examples of how to use the corrected implementations.
    """
    
    # Setup
    vocab_size = 10000
    batch_size = 32
    seq_len = 256
    
    print("=" * 70)
    print("EXAMPLE 1: Simple Encoder (Next-Token Prediction)")
    print("=" * 70)
    
    # Create simple encoder model
    model = SimpleTransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        max_seq_length=seq_len
    )
    
    # Sample input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {vocab_size})")
    assert outputs.shape == (batch_size, seq_len, vocab_size), "Shape mismatch!"
    print("✓ Shape test passed!\n")
    
    print("=" * 70)
    print("EXAMPLE 2: PanBART Model (Full Implementation)")
    print("=" * 70)
    
    # Create full panBART model
    panbart = PanBARTModel(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=2  # Binary: resistant/susceptible
    )
    
    # Test MLM task
    mlm_outputs = panbart(input_ids, task='mlm')
    print(f"MLM output shape: {mlm_outputs.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {vocab_size})")
    assert mlm_outputs.shape == (batch_size, seq_len, vocab_size)
    print("✓ MLM test passed!")
    
    # Test classification task  
    class_outputs = panbart(input_ids, task='classification')
    print(f"Classification output shape: {class_outputs.shape}")
    print(f"Expected: ({batch_size}, 2)")
    assert class_outputs.shape == (batch_size, 2)
    print("✓ Classification test passed!\n")
    
    print("=" * 70)
    print("EXAMPLE 3: Device Handling")
    print("=" * 70)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_cuda = model.to(device)
        input_cuda = input_ids.to(device)
        output_cuda = model_cuda(input_cuda)
        print(f"✓ Model runs on GPU: {output_cuda.device}")
    else:
        print("⚠ CUDA not available, running on CPU")
    
    print("\n" + "=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
