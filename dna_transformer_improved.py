"""
Improved DNA Sequence Processing and Transformer Model
Supports: K-mers, BPE tokenization, One-hot encoding, and Transformer architecture
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple, Dict, Union
import gzip
from pathlib import Path
import numpy as np
from collections import Counter
import json
from Bio import SeqIO
from Bio.Seq import Seq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers


# ============================================================================
# 1. IMPROVED FASTA READING
# ============================================================================

class FASTAReader:
    """Robust FASTA file reader with support for gzipped files"""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.is_gzipped = self.file_path.suffix == '.gz'
    
    def read_sequences(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Read sequences from FASTA file
        
        Args:
            limit: Maximum number of sequences to read
            
        Returns:
            List of dictionaries with 'id', 'description', 'sequence'
        """
        sequences = []
        
        try:
            if self.is_gzipped:
                handle = gzip.open(self.file_path, 'rt')
            else:
                handle = open(self.file_path, 'r')
            
            for idx, record in enumerate(SeqIO.parse(handle, "fasta")):
                if limit and idx >= limit:
                    break
                
                # Parse ID (e.g., V240_1)
                seq_id = record.id
                parts = seq_id.split('_')
                
                sequence_data = {
                    'id': seq_id,
                    'v_number': parts[0] if len(parts) > 0 else '',
                    'sample_number': parts[1] if len(parts) > 1 else '',
                    'description': record.description,
                    'sequence': str(record.seq),
                    'length': len(record.seq)
                }
                
                sequences.append(sequence_data)
            
            handle.close()
            
        except Exception as e:
            raise IOError(f"Error reading FASTA file: {e}")
        
        return sequences
    
    def read_sequences_as_strings(self, limit: Optional[int] = None) -> List[str]:
        """Read sequences and return only the sequence strings"""
        sequences = self.read_sequences(limit=limit)
        return [seq['sequence'] for seq in sequences]
    
    def get_sequence_stats(self) -> Dict:
        """Get statistics about sequences in the file"""
        sequences = self.read_sequences()
        lengths = [seq['length'] for seq in sequences]
        
        stats = {
            'total_sequences': len(sequences),
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'mean_length': np.mean(lengths) if lengths else 0,
            'median_length': np.median(lengths) if lengths else 0
        }
        
        return stats


# ============================================================================
# 2. K-MER TOKENIZATION
# ============================================================================

class KmerTokenizer:
    """K-mer based tokenization for DNA sequences"""
    
    def __init__(self, k: int = 3, stride: int = 1):
        """
        Args:
            k: Length of k-mers
            stride: Step size between k-mers (1=overlapping, k=non-overlapping)
        """
        self.k = k
        self.stride = stride
        self.vocab = {}
        self.id_to_token = {}
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[GAP]', '[BOS]', '[EOS]']
        
        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
    
    def build_vocab(self, sequences: List[str], max_vocab_size: Optional[int] = None):
        """Build vocabulary from sequences"""
        kmer_counts = Counter()
        
        for seq in sequences:
            kmers = self.sequence_to_kmers(seq)
            kmer_counts.update(kmers)
        
        # Add k-mers to vocabulary
        start_idx = len(self.special_tokens)
        
        # Sort by frequency
        most_common = kmer_counts.most_common(max_vocab_size) if max_vocab_size else kmer_counts.items()
        
        for idx, (kmer, _) in enumerate(most_common, start=start_idx):
            if kmer not in self.vocab:
                self.vocab[kmer] = idx
                self.id_to_token[idx] = kmer
    
    def sequence_to_kmers(self, sequence: str) -> List[str]:
        """Convert sequence to k-mers"""
        sequence = sequence.upper()
        kmers = []
        for i in range(0, len(sequence) - self.k + 1, self.stride):
            kmers.append(sequence[i:i+self.k])
        return kmers
    
    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """Convert sequence to token IDs"""
        kmers = self.sequence_to_kmers(sequence)
        
        ids = []
        if add_special_tokens:
            ids.append(self.vocab['[CLS]'])
        
        for kmer in kmers:
            ids.append(self.vocab.get(kmer, self.vocab['[UNK]']))
        
        if add_special_tokens:
            ids.append(self.vocab['[SEP]'])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to sequence (approximate for overlapping k-mers)"""
        tokens = [self.id_to_token.get(id, '[UNK]') for id in ids]
        
        # Filter out special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        
        if not tokens:
            return ""
        
        # Reconstruct sequence
        if self.stride == self.k:
            # Non-overlapping: just concatenate
            return ''.join(tokens)
        else:
            # Overlapping: use first k-mer fully, then add last char of each subsequent k-mer
            sequence = tokens[0]
            for kmer in tokens[1:]:
                if len(kmer) == self.k:
                    sequence += kmer[-1]
            return sequence
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'k': self.k,
                'stride': self.stride
            }, f, indent=2)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.k = data['k']
            self.stride = data['stride']
            self.id_to_token = {int(v): k for k, v in self.vocab.items()}


# ============================================================================
# 3. BPE TOKENIZATION
# ============================================================================

class DNABPETokenizer:
    """Byte-Pair Encoding tokenizer for DNA sequences"""
    
    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None
    
    def train(self, sequences: List[str]):
        """Train BPE tokenizer on DNA sequences"""
        # Initialize tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Add normalizer for DNA (uppercase)
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.Lowercase()
        ])
        
        # Pre-tokenizer: character-level for DNA
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit()
        ])
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
        )
        
        # Train on sequences
        self.tokenizer.train_from_iterator(sequences, trainer)
    
    def encode(self, sequence: str) -> List[int]:
        """Encode sequence to token IDs"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        encoding = self.tokenizer.encode(sequence)
        return encoding.ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to sequence"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        return self.tokenizer.decode(ids)
    
    def save(self, path: str):
        """Save tokenizer to file"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        self.tokenizer.save(path)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        self.tokenizer = Tokenizer.from_file(path)


# ============================================================================
# 4. ONE-HOT ENCODING
# ============================================================================

class OneHotEncoder:
    """One-hot encoding for DNA sequences"""
    
    def __init__(self, include_ambiguous: bool = True):
        """
        Args:
            include_ambiguous: Whether to handle ambiguous bases (N, R, Y, etc.)
        """
        self.include_ambiguous = include_ambiguous
        
        # Standard bases
        self.base_mapping = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1]
        }
        
        if include_ambiguous:
            # Ambiguous bases (use all zeros or custom encoding)
            self.base_mapping.update({
                'N': [0, 0, 0, 0],  # Unknown
                'R': [0.5, 0, 0.5, 0],  # A or G
                'Y': [0, 0.5, 0, 0.5],  # C or T
                'S': [0, 0.5, 0.5, 0],  # G or C
                'W': [0.5, 0, 0, 0.5],  # A or T
                'K': [0, 0, 0.5, 0.5],  # G or T
                'M': [0.5, 0.5, 0, 0],  # A or C
                'B': [0, 0.33, 0.33, 0.33],  # C or G or T
                'D': [0.33, 0, 0.33, 0.33],  # A or G or T
                'H': [0.33, 0.33, 0, 0.33],  # A or C or T
                'V': [0.33, 0.33, 0.33, 0],  # A or C or G
            })
    
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode DNA sequence to one-hot representation
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            numpy array of shape (seq_length, 4)
        """
        sequence = sequence.upper()
        encoded = []
        
        for base in sequence:
            if base in self.base_mapping:
                encoded.append(self.base_mapping[base])
            else:
                # Unknown base
                encoded.append([0, 0, 0, 0])
        
        return np.array(encoded, dtype=np.float32)
    
    def encode_tensor(self, sequence: str) -> torch.Tensor:
        """Encode sequence and return PyTorch tensor"""
        encoded = self.encode(sequence)
        return torch.tensor(encoded, dtype=torch.float32)
    
    def decode(self, one_hot: np.ndarray) -> str:
        """
        Decode one-hot representation to sequence
        
        Args:
            one_hot: numpy array of shape (seq_length, 4)
            
        Returns:
            DNA sequence string
        """
        bases = ['A', 'C', 'G', 'T']
        sequence = []
        
        for vec in one_hot:
            idx = np.argmax(vec)
            if vec[idx] > 0:
                sequence.append(bases[idx])
            else:
                sequence.append('N')
        
        return ''.join(sequence)


# ============================================================================
# 5. DATASET FOR TRANSFORMER
# ============================================================================

class DNASequenceDataset(Dataset):
    """Dataset for DNA sequences with masked language modeling"""
    
    def __init__(
        self,
        sequences: List[str],
        tokenizer: Union[KmerTokenizer, DNABPETokenizer],
        max_length: int = 512,
        mask_prob: float = 0.15,
        encoding_type: str = 'kmer'  # 'kmer', 'bpe', or 'onehot'
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.encoding_type = encoding_type
        
        if encoding_type == 'onehot':
            self.onehot_encoder = OneHotEncoder()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        if self.encoding_type == 'onehot':
            # One-hot encoding
            encoded = self.onehot_encoder.encode_tensor(sequence)
            # Truncate if needed
            if len(encoded) > self.max_length:
                encoded = encoded[:self.max_length]
            return {
                'input': encoded,
                'length': len(encoded)
            }
        else:
            # Token-based encoding (k-mer or BPE)
            tokens = self.tokenizer.encode(sequence)
            
            # Truncate if needed
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            
            # Create masked version for MLM
            masked_tokens, labels = self._create_masked_lm_predictions(tokens)
            
            return {
                'input_ids': torch.tensor(masked_tokens, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'length': len(tokens)
            }
    
    def _create_masked_lm_predictions(self, tokens: List[int]) -> Tuple[List[int], List[int]]:
        """Create masked tokens for MLM"""
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 is ignored by CrossEntropyLoss
        
        mask_token_id = self.tokenizer.vocab.get('[MASK]', 0)
        
        for i in range(len(tokens)):
            if np.random.random() < self.mask_prob:
                labels[i] = tokens[i]
                masked_tokens[i] = mask_token_id
        
        return masked_tokens, labels


def collate_fn_tokens(batch):
    """Collate function for token-based batches"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }


def collate_fn_onehot(batch):
    """Collate function for one-hot encoded batches"""
    inputs = [item['input'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    # Pad sequences
    max_len = max(lengths)
    batch_size = len(inputs)
    
    padded = torch.zeros(batch_size, max_len, 4)
    for i, inp in enumerate(inputs):
        padded[i, :len(inp)] = inp
    
    return {
        'input': padded,
        'lengths': torch.tensor(lengths)
    }


# ============================================================================
# 6. TRANSFORMER MODEL
# ============================================================================

class DNATransformerEncoder(nn.Module):
    """Transformer encoder for DNA sequences"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (batch_size, seq_len)
            src_mask: (seq_len, seq_len)
            src_key_padding_mask: (batch_size, seq_len)
        """
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        logits = self.output_layer(output)
        return logits


class DNATransformerOneHot(nn.Module):
    """Transformer for one-hot encoded DNA sequences"""
    
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 4
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, num_classes)
    
    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: (batch_size, seq_len, 4)
            src_key_padding_mask: (batch_size, seq_len)
        """
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=src_key_padding_mask
        )
        
        logits = self.output_layer(output)
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        return self.dropout(x)


# ============================================================================
# 7. TRAINING UTILITIES
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, encoding_type='kmer'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        if encoding_type == 'onehot':
            inputs = batch['input'].to(device)
            # For autoencoder-style training
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 4), inputs.view(-1, 4))
        else:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, encoding_type='kmer'):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if encoding_type == 'onehot':
                inputs = batch['input'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 4), inputs.view(-1, 4))
            else:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ============================================================================
# 8. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("DNA Transformer - Example Usage")
    print("=" * 50)
    
    # Example: Read FASTA file
    # fasta_reader = FASTAReader("path/to/your/file.fasta.gz")
    # sequences_data = fasta_reader.read_sequences(limit=100)
    # sequences = [seq['sequence'] for seq in sequences_data]
    
    # For demonstration, use example sequences
    sequences = [
        "ATGCTAGCTAGCTAGCTAGCTAATGCTAGC",
        "GGCTACGTTACGACGTAACGTA",
        "TTACTGACCTGAACCTGACCTA",
        "ACGTACGTACGTACGTACGTAC",
    ]
    
    print(f"\nLoaded {len(sequences)} sequences")
    
    # Example 1: K-mer tokenization
    print("\n1. K-mer Tokenization (k=3)")
    print("-" * 50)
    kmer_tokenizer = KmerTokenizer(k=3, stride=1)
    kmer_tokenizer.build_vocab(sequences)
    print(f"Vocabulary size: {len(kmer_tokenizer.vocab)}")
    encoded = kmer_tokenizer.encode(sequences[0])
    print(f"Encoded: {encoded[:10]}...")
    decoded = kmer_tokenizer.decode(encoded)
    print(f"Decoded: {decoded[:30]}...")
    
    # Example 2: BPE tokenization
    print("\n2. BPE Tokenization")
    print("-" * 50)
    bpe_tokenizer = DNABPETokenizer(vocab_size=500)
    bpe_tokenizer.train(sequences)
    encoded_bpe = bpe_tokenizer.encode(sequences[0])
    print(f"Encoded: {encoded_bpe[:10]}...")
    decoded_bpe = bpe_tokenizer.decode(encoded_bpe)
    print(f"Decoded: {decoded_bpe[:30]}...")
    
    # Example 3: One-hot encoding
    print("\n3. One-Hot Encoding")
    print("-" * 50)
    onehot_encoder = OneHotEncoder()
    encoded_onehot = onehot_encoder.encode(sequences[0][:10])
    print(f"Shape: {encoded_onehot.shape}")
    print(f"First base encoding: {encoded_onehot[0]}")
    
    print("\n" + "=" * 50)
    print("Setup complete! Ready to train transformer models.")
