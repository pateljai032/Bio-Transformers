"""
Complete Training Example for DNA Transformer
Demonstrates K-mer, BPE, and One-hot encoding approaches
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dna_transformer_improved import (
    FASTAReader,
    KmerTokenizer,
    DNABPETokenizer,
    OneHotEncoder,
    DNASequenceDataset,
    DNATransformerEncoder,
    DNATransformerOneHot,
    collate_fn_tokens,
    collate_fn_onehot,
    train_epoch,
    evaluate
)


def main():
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # File paths
    FASTA_FILE = "path/to/your/file.fasta.gz"  # Replace with your actual file
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    MAX_SEQ_LENGTH = 512
    
    # Model parameters
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    
    # Encoding method: 'kmer', 'bpe', or 'onehot'
    ENCODING_TYPE = 'kmer'  # Change this to test different methods
    
    # K-mer settings (if using k-mer encoding)
    K = 6
    STRIDE = 3  # 1 for overlapping, K for non-overlapping
    
    # BPE settings (if using BPE encoding)
    BPE_VOCAB_SIZE = 1000
    BPE_MIN_FREQ = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Encoding method: {ENCODING_TYPE}")
    print("=" * 70)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    
    print("\n1. Loading FASTA data...")
    print("-" * 70)
    
    # Option 1: Load from real FASTA file
    # fasta_reader = FASTAReader(FASTA_FILE)
    # sequences_data = fasta_reader.read_sequences(limit=1000)
    # sequences = [seq['sequence'] for seq in sequences_data]
    # print(f"Loaded {len(sequences)} sequences")
    # stats = fasta_reader.get_sequence_stats()
    # print(f"Stats: {stats}")
    
    # Option 2: Use example sequences (for demonstration)
    sequences = [
        "ATGCTAGCTAGCTAGCTAGCTAATGCTAGCGATCGATCGTAGCTAGCTGATCGATCG",
        "GGCTACGTTACGACGTAACGTACGATCGATCGATCGTAGCTAGCTACGATCGATCGA",
        "TTACTGACCTGAACCTGACCTACGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCT",
        "ACGTACGTACGTACGTACGTACGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTA",
        "ATGCGGATCCGATCGATCGATCGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTA",
        "GGCATGCTAGCATCGATGCATGCGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCT",
        "TTACGATCGATCGTGCACGATCGATCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTA",
        "CGATCGATCGATCGTAGCTAGCTACGATCGATCGATCGATCGATCGATCGTAGCTAG",
    ] * 10  # Replicate for more training data
    
    print(f"Using {len(sequences)} sequences for training")
    
    # ========================================================================
    # PREPARE TOKENIZER/ENCODER
    # ========================================================================
    
    print(f"\n2. Preparing {ENCODING_TYPE.upper()} tokenizer/encoder...")
    print("-" * 70)
    
    if ENCODING_TYPE == 'kmer':
        # K-mer tokenization
        tokenizer = KmerTokenizer(k=K, stride=STRIDE)
        tokenizer.build_vocab(sequences)
        vocab_size = len(tokenizer.vocab)
        print(f"K-mer vocabulary size: {vocab_size}")
        print(f"K: {K}, Stride: {STRIDE}")
        print(f"Sample k-mers: {list(tokenizer.vocab.keys())[8:18]}")
        
    elif ENCODING_TYPE == 'bpe':
        # BPE tokenization
        tokenizer = DNABPETokenizer(vocab_size=BPE_VOCAB_SIZE, min_frequency=BPE_MIN_FREQ)
        tokenizer.train(sequences)
        vocab_size = BPE_VOCAB_SIZE
        print(f"BPE vocabulary size: {vocab_size}")
        
    else:  # onehot
        # One-hot encoding
        tokenizer = None
        vocab_size = 4
        print("One-hot encoding: 4 dimensions (A, C, G, T)")
    
    # ========================================================================
    # CREATE DATASET AND DATALOADER
    # ========================================================================
    
    print("\n3. Creating dataset and dataloader...")
    print("-" * 70)
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    
    train_dataset = DNASequenceDataset(
        train_sequences,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        encoding_type=ENCODING_TYPE
    )
    
    val_dataset = DNASequenceDataset(
        val_sequences,
        tokenizer,
        max_length=MAX_SEQ_LENGTH,
        encoding_type=ENCODING_TYPE
    )
    
    # Choose appropriate collate function
    collate_fn = collate_fn_onehot if ENCODING_TYPE == 'onehot' else collate_fn_tokens
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    
    print("\n4. Creating model...")
    print("-" * 70)
    
    if ENCODING_TYPE == 'onehot':
        model = DNATransformerOneHot(
            input_dim=4,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            num_classes=4
        )
    else:
        model = DNATransformerEncoder(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            max_seq_length=MAX_SEQ_LENGTH
        )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # ========================================================================
    # TRAINING SETUP
    # ========================================================================
    
    print("\n5. Setting up training...")
    print("-" * 70)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n6. Training model...")
    print("=" * 70)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 70)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, ENCODING_TYPE
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate(
            model, val_loader, criterion, device, ENCODING_TYPE
        )
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'best_model_{ENCODING_TYPE}.pt')
            print("✓ Saved best model")
    
    # ========================================================================
    # PLOT TRAINING CURVES
    # ========================================================================
    
    print("\n7. Plotting training curves...")
    print("-" * 70)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves - {ENCODING_TYPE.upper()} Encoding')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'training_curves_{ENCODING_TYPE}.png', dpi=300)
    print(f"Saved plot: training_curves_{ENCODING_TYPE}.png")
    
    # ========================================================================
    # SAVE TOKENIZER
    # ========================================================================
    
    print("\n8. Saving tokenizer/encoder...")
    print("-" * 70)
    
    if ENCODING_TYPE == 'kmer':
        tokenizer.save_vocab(f'kmer_vocab_k{K}.json')
        print(f"Saved: kmer_vocab_k{K}.json")
    elif ENCODING_TYPE == 'bpe':
        tokenizer.save(f'bpe_tokenizer_{BPE_VOCAB_SIZE}.json')
        print(f"Saved: bpe_tokenizer_{BPE_VOCAB_SIZE}.json")
    
    # ========================================================================
    # TESTING PREDICTIONS
    # ========================================================================
    
    print("\n9. Testing predictions...")
    print("-" * 70)
    
    model.eval()
    test_seq = sequences[0][:100]  # Use first 100 bases
    
    if ENCODING_TYPE == 'onehot':
        encoder = OneHotEncoder()
        test_input = encoder.encode_tensor(test_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(test_input)
        predicted = torch.argmax(output, dim=-1)[0]
        bases = ['A', 'C', 'G', 'T']
        predicted_seq = ''.join([bases[i] for i in predicted.cpu().numpy()])
    else:
        test_input = torch.tensor(tokenizer.encode(test_seq)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(test_input)
        predicted = torch.argmax(output, dim=-1)[0]
        predicted_seq = tokenizer.decode(predicted.cpu().tolist())
    
    print(f"Input sequence:     {test_seq[:50]}...")
    print(f"Predicted sequence: {predicted_seq[:50]}...")
    
    # Calculate accuracy
    min_len = min(len(test_seq), len(predicted_seq))
    accuracy = sum(1 for i in range(min_len) if test_seq[i] == predicted_seq[i]) / min_len
    print(f"Character-level accuracy: {accuracy:.2%}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: best_model_{ENCODING_TYPE}.pt")


if __name__ == "__main__":
    main()
