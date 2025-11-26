# DNA Transformer - Improved Implementation

A comprehensive implementation of transformer models for DNA sequence analysis with support for multiple encoding methods: K-mers, BPE (Byte-Pair Encoding), and One-hot encoding.

## Features

**Robust FASTA Reading**
- Support for gzipped and regular FASTA files
- Automatic sequence ID parsing (e.g., V240_1 format)
- Sequence statistics and batch processing

**Multiple Encoding Methods**
- **K-mer tokenization**: Overlapping or non-overlapping k-mers
- **BPE tokenization**: Learned subword units for DNA
- **One-hot encoding**: Direct base-level encoding with ambiguous base support

**Transformer Architecture**
- Encoder-only transformer for masked language modeling
- Support for both token-based and one-hot encoded inputs
- Positional encoding and attention mechanisms

**Production-Ready**
- Comprehensive error handling
- Vocabulary saving/loading
- Training/validation split
- Learning rate scheduling
- Model checkpointing

## Installation

```bash
pip install torch torchvision torchaudio
pip install biopython
pip install tokenizers
pip install numpy matplotlib
```

## Quick Start

### 1. Reading FASTA Files

```python
from dna_transformer_improved import FASTAReader

# Read from gzipped or regular FASTA file
reader = FASTAReader("your_file.fasta.gz")

# Get all sequences with metadata
sequences_data = reader.read_sequences(limit=100)
for seq in sequences_data:
    print(f"ID: {seq['id']}")
    print(f"V Number: {seq['v_number']}")
    print(f"Sample: {seq['sample_number']}")
    print(f"Length: {seq['length']}")
    print(f"Sequence: {seq['sequence'][:50]}...")

# Or just get sequence strings
sequences = reader.read_sequences_as_strings(limit=100)

# Get statistics
stats = reader.get_sequence_stats()
print(stats)
# Output: {'total_sequences': 10, 'min_length': 1000, 'max_length': 500000, ...}
```

### 2. K-mer Tokenization

```python
from dna_transformer_improved import KmerTokenizer

# Create tokenizer
tokenizer = KmerTokenizer(k=6, stride=3)  # k=6 for 6-mers, stride=3 for overlapping

# Build vocabulary from sequences
tokenizer.build_vocab(sequences)
print(f"Vocabulary size: {len(tokenizer.vocab)}")

# Encode sequence
sequence = "ATGCTAGCTAGCTA"
encoded = tokenizer.encode(sequence)
print(f"Encoded: {encoded}")

# Decode back to sequence
decoded = tokenizer.decode(encoded)
print(f"Decoded: {decoded}")

# Save vocabulary
tokenizer.save_vocab("kmer_vocab.json")

# Load vocabulary
tokenizer.load_vocab("kmer_vocab.json")
```

**K-mer Parameters:**
- `k=3`: Small k-mers, large vocabulary, captures local patterns
- `k=6`: Medium k-mers, balanced vocabulary
- `k=9`: Large k-mers, smaller vocabulary, captures longer motifs
- `stride=1`: Overlapping k-mers (more data)
- `stride=k`: Non-overlapping k-mers (faster, less data)

### 3. BPE Tokenization

```python
from dna_transformer_improved import DNABPETokenizer

# Create and train BPE tokenizer
tokenizer = DNABPETokenizer(vocab_size=1000, min_frequency=2)
tokenizer.train(sequences)

# Encode/decode
encoded = tokenizer.encode("ATGCTAGCTA")
decoded = tokenizer.decode(encoded)

# Save/load
tokenizer.save("bpe_tokenizer.json")
tokenizer.load("bpe_tokenizer.json")
```

### 4. One-Hot Encoding

```python
from dna_transformer_improved import OneHotEncoder

# Create encoder
encoder = OneHotEncoder(include_ambiguous=True)

# Encode sequence
sequence = "ATGCN"
encoded = encoder.encode(sequence)  # Returns numpy array (5, 4)
print(encoded)
# [[1, 0, 0, 0],  # A
#  [0, 0, 0, 1],  # T
#  [0, 0, 1, 0],  # G
#  [0, 1, 0, 0],  # C
#  [0, 0, 0, 0]]  # N

# Encode as PyTorch tensor
tensor = encoder.encode_tensor(sequence)

# Decode back
decoded = encoder.decode(encoded)
print(decoded)  # "ATGCN"
```

**Ambiguous Base Handling:**
- `N`: Unknown (all zeros)
- `R`: A or G (50% each)
- `Y`: C or T (50% each)
- `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`: Other IUPAC codes

### 5. Training a Transformer Model

```python
import torch
from torch.utils.data import DataLoader
from dna_transformer_improved import (
    KmerTokenizer,
    DNASequenceDataset,
    DNATransformerEncoder,
    collate_fn_tokens,
    train_epoch,
    evaluate
)

# Prepare tokenizer
tokenizer = KmerTokenizer(k=6, stride=3)
tokenizer.build_vocab(sequences)

# Create dataset
dataset = DNASequenceDataset(
    sequences,
    tokenizer,
    max_length=512,
    encoding_type='kmer'
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn_tokens
)

# Create model
model = DNATransformerEncoder(
    vocab_size=len(tokenizer.vocab),
    d_model=256,
    nhead=8,
    num_layers=4
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(10):
    train_loss = train_epoch(model, dataloader, optimizer, criterion, device, 'kmer')
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")
```

### 6. Complete Training Pipeline

Use the provided training script:

```python
# Edit train_dna_transformer.py to set your parameters:
# - FASTA_FILE: Path to your FASTA file
# - ENCODING_TYPE: 'kmer', 'bpe', or 'onehot'
# - K, STRIDE: K-mer parameters
# - Model hyperparameters

python train_dna_transformer.py
```

## Architecture Details

### Transformer Encoder (for K-mer/BPE)
```
Input (token IDs) 
  → Embedding Layer
  → Positional Encoding
  → Transformer Encoder (Multi-head Attention + FFN) × N layers
  → Output Projection
  → Vocabulary Logits
```

### Transformer for One-Hot Encoding
```
Input (one-hot vectors)
  → Linear Projection to d_model
  → Positional Encoding
  → Transformer Encoder × N layers
  → Output Projection
  → Base Logits (4 classes: A, C, G, T)
```

## Model Hyperparameters

| Parameter | Recommended Values | Description |
|-----------|-------------------|-------------|
| `d_model` | 256, 512 | Model dimension |
| `nhead` | 8, 16 | Number of attention heads |
| `num_layers` | 4, 6, 8 | Number of transformer layers |
| `dim_feedforward` | 1024, 2048 | FFN hidden dimension |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_length` | 512, 1024 | Maximum sequence length |

## Choosing an Encoding Method

### K-mer Tokenization
**Pros:**
- Fast and efficient
- Captures local sequence patterns
- Easy to interpret
- Good for motif discovery

**Cons:**
- Fixed vocabulary size (4^k possible k-mers)
- Doesn't adapt to data
- May miss longer-range dependencies

**Best for:** Genomic sequences, motif analysis, local pattern recognition

### BPE Tokenization
**Pros:**
- Learns optimal subword units
- Flexible vocabulary
- Handles rare patterns well
- Good compression

**Cons:**
- Requires training
- Less interpretable
- Slower than k-mers

**Best for:** Large diverse datasets, transfer learning, compression

### One-Hot Encoding
**Pros:**
- Simple and direct
- No tokenization needed
- Preserves base-level information
- Good for small sequences

**Cons:**
- High memory usage
- No sequence compression
- Limited context window
- Slower training

**Best for:** Short sequences, base-level predictions, regulatory elements

## Performance Tips

1. **For long sequences (>10kb):**
   - Use k-mers with k=6-9
   - Set stride=3-6 for compression
   - Use gradient checkpointing

2. **For short sequences (<1kb):**
   - Use k-mers with k=3-6
   - Or use one-hot encoding
   - Smaller model (d_model=256)

3. **For large datasets:**
   - Use BPE for vocabulary compression
   - Train with mixed precision
   - Use data parallelism

4. **Memory optimization:**
   - Reduce batch size
   - Reduce max_seq_length
   - Use gradient accumulation
   - Enable gradient checkpointing

## Example: Full Pipeline

```python
from dna_transformer_improved import *
import torch
from torch.utils.data import DataLoader

# 1. Load data
reader = FASTAReader("genome.fasta.gz")
sequences = reader.read_sequences_as_strings(limit=1000)

# 2. Choose encoding
tokenizer = KmerTokenizer(k=6, stride=3)
tokenizer.build_vocab(sequences)

# 3. Create dataset
train_dataset = DNASequenceDataset(
    sequences[:800],
    tokenizer,
    max_length=512,
    encoding_type='kmer'
)

val_dataset = DNASequenceDataset(
    sequences[800:],
    tokenizer,
    max_length=512,
    encoding_type='kmer'
)

# 4. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_tokens)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_tokens)

# 5. Create model
model = DNATransformerEncoder(
    vocab_size=len(tokenizer.vocab),
    d_model=256,
    nhead=8,
    num_layers=4
)

# 6. Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 'kmer')
    val_loss = evaluate(model, val_loader, criterion, device, 'kmer')
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# 7. Save model
torch.save(model.state_dict(), "dna_transformer.pt")
tokenizer.save_vocab("tokenizer.json")
```

## File Structure

```
.
├── dna_transformer_improved.py   # Main implementation
├── train_dna_transformer.py      # Training script
├── README.md                      # This file
└── your_data.fasta.gz            # Your FASTA file
```

## Common Issues and Solutions

### Issue: Out of memory during training
**Solution:** Reduce batch size, reduce max_seq_length, or use gradient accumulation

### Issue: Vocabulary too large with k-mers
**Solution:** Use larger stride, smaller k, or switch to BPE

### Issue: Training loss not decreasing
**Solution:** Check learning rate, increase model capacity, verify data loading

### Issue: Slow training
**Solution:** Use GPU, reduce sequence length, use larger stride with k-mers

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dna_transformer_improved,
  title = {DNA Transformer: Improved Implementation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/dna-transformer}
}
```

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
