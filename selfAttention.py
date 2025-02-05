class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by number of heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, query, key, value, mask):
        N = query.shape[0]  # Batch size
        seq_length = query.shape[1]
        
        query = query.view(N, seq_length, self.heads, self.head_dim)
        key = key.view(N, seq_length, self.heads, self.head_dim)
        value = value.view(N, seq_length, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, value]).reshape(N, seq_length, self.embed_size)
        out = self.fc_out(out)
        return out