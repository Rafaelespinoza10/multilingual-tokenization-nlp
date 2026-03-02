import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LanguageClassifier(nn.Module):
    """
    Neural network model for language classification.

    Supports:
    - mean pooling over token embeddings (recommended)
    - flattening embeddings to a fixed-size vector (requires fixed seq_len)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int = 3,   # es, en, fr
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        use_mean_pooling: bool = True,
        max_sequence_length: int = 100, 
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.use_mean_pooling = use_mean_pooling
        self.max_sequence_length = max_sequence_length

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        if self.use_mean_pooling:
            input_dim = embedding_dim
        else:
            input_dim = embedding_dim * max_sequence_length

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.hidden_layers = nn.Sequential(*layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                # Optional: zero-out pad embedding after init
                if self.padding_idx is not None:
                    with torch.no_grad():
                        self.embedding.weight[self.padding_idx].fill_(0.0)

    def _mean_pool(self, embeddings: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, T, E]
        input_ids:  [B, T]
        returns:    [B, E]
        """
        mask = (input_ids != self.padding_idx).unsqueeze(-1)  # [B, T, 1]
        masked_embeddings = embeddings * mask
        lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = masked_embeddings.sum(dim=1) / lengths
        return pooled

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        returns logits: [batch_size, num_classes]
        """
        embeddings = self.embedding(input_ids)  # [B, T, E]

        if self.use_mean_pooling:
            features = self._mean_pool(embeddings, input_ids)  # [B, E]
        else:
            # Requires fixed sequence length == max_sequence_length
            batch_size, seq_len, emb_dim = embeddings.shape
            if seq_len != self.max_sequence_length:
                raise ValueError(
                    f"Expected seq_len={self.max_sequence_length} for flatten mode, got {seq_len}. "
                    "Use mean pooling or pad/truncate to a fixed length."
                )
            features = embeddings.reshape(batch_size, -1)  # [B, T*E]

        hidden = self.hidden_layers(features)
        logits = self.classifier(hidden)
        return logits

    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        logits = self.forward(input_ids)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.forward(input_ids)
        return F.softmax(logits, dim=-1)
    

class LanguageClassifierTrainer:
    """Trainer for the language classifier."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_path: str = "artifacts/best_model.pth",
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Train for one epoch and return (loss, accuracy)."""
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty")

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            input_ids = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

        average_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0
        return average_loss, accuracy

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validate model and return (loss, accuracy)."""
        if len(val_loader) == 0:
            raise ValueError("val_loader is empty")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_ids)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        average_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        return average_loss, accuracy

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 20,
        early_stopping: int = 5,
    ):
        """Train the model with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 20)

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            print(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"Model saved -> {self.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        print("Loaded best model")