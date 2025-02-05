class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, dropout, forward_expansion, input_length):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, 2)  # Predicting (Vx, Vy)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        out = self.fc_out(x[:, -1, :])  # Use the last time step for prediction
        return out
