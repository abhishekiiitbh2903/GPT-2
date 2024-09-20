from dataclasses import dataclass

# class GPTConfig:
#     def __init__(self, block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768):
#         self.block_size = block_size
#         self.vocab_size = vocab_size
#         self.n_layer = n_layer
#         self.n_head = n_head
#         self.n_embd = n_embd

#     def __repr__(self):
#         return (f"GPTConfig(block_size={self.block_size}, vocab_size={self.vocab_size}, "
#                 f"n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd})")

####################################### Easier Way ###########################################

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
