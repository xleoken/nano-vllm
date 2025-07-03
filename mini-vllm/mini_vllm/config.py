class ModelConfig:
    def __init__(self, vocab_size=64000, hidden_size=2048, num_layers=24,
                 num_heads=32, head_dim=64, max_seq_len=2048):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.intermediate_size = hidden_size * 4    