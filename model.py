import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_curve

class FinancialDataset(Dataset):
    def __init__(self, market_data: pd.DataFrame, sentiment_data: pd.DataFrame):
        self.market_data = market_data
        self.sentiment_data = sentiment_data
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._align_data()

    def _align_data(self):
        """Align timestamps between market and sentiment data"""
        combined_index = self.market_data.index.union(self.sentiment_data.index)
        self.market_data = self.market_data.reindex(combined_index).ffill()
        self.sentiment_data = self.sentiment_data.reindex(combined_index).ffill()

    def __len__(self):
        return len(self.market_data) - 60  # Use 60-period lookback

    def __getitem__(self, idx):
        market_window = self.market_data.iloc[idx:idx+60]
        sentiment_text = self.sentiment_data.iloc[idx+60]['text']
        
        inputs = self.tokenizer(
            sentiment_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        market_tensor = torch.FloatTensor(market_window.values)
        target = torch.FloatTensor([self.market_data.iloc[idx+60]['returns']])
        
        return {
            'text_inputs': inputs,
            'market_data': market_tensor,
            'target': target
        }

class MultiModalFusion(nn.Module):
    """Enhanced neural architecture with attention mechanisms"""
    def __init__(self, text_dim: int = 768, market_dim: int = 20, hidden_dim: int = 256):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.market_encoder = MarketEncoder(input_dim=market_dim, hidden_dim=hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.temporal_conv = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3)
        self.output = nn.Linear(hidden_dim, 1)
        
    def forward(self, text_inputs: Dict, market_data: torch.Tensor) -> torch.Tensor:
        # Text pathway
        text_features = self.text_encoder(**text_inputs).last_hidden_state[:, 0, :]
        
        # Market pathway
        market_features = self.market_encoder(market_data)
        
        # Attention-based fusion
        attn_output, _ = self.attention(
            text_features.unsqueeze(0),
            market_features.unsqueeze(0),
            market_features.unsqueeze(0)
        )
        
        # Temporal convolution
        combined = torch.cat([text_features, attn_output.squeeze(0)], dim=1)
        fused = self.fusion(combined)
        conv_out = self.temporal_conv(fused.unsqueeze(0).transpose(1,2))
        
        return self.output(conv_out.squeeze().transpose(0,1)[-1])

class MarketEncoder(nn.Module):
    """LSTM-based market data encoder with skip connections"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.ReLU()
            ) for _ in range(2)
        ])
        
    def forward(self, x):
        out, _ = self.lstm(x)
        for block in self.res_blocks:
            out = out + block(out)
        return out[:, -1, :]

def train_model(model: nn.Module, dataloader: DataLoader, epochs: int = 50):
    """Complete training loop with early stopping"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch['text_inputs'], batch['market_data'])
            loss = criterion(outputs, batch['target'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

