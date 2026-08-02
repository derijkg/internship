# features/dense_encoder.py

from typing import List
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def get_optimal_device(requested_device: str = "cpu") -> torch.device:
    """
    Safely determines the execution device.
    """
    if "cuda" in requested_device and torch.cuda.is_available():
        return torch.device(requested_device)
    return torch.device("cpu")

class DenseTransformerEncoder(nn.Module):
    """
    Extracts dense contextual representations for sentence sequences using a 
    Dutch Transformer model (e.g. NFI RobBERT Sentence Transformer).
    """
    def __init__(
        self,
        model_name: str = "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
        device: str = "cpu",
        max_length: int = 128
    ):
        super().__init__()
        self.device = get_optimal_device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.hidden_dim = 768  # Standard RoBERTa-base hidden size

        print(f"Loading Pretrained Dutch Transformer '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()

        # Freeze backbone parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def extract_sentence_embeddings(self, sents: List[str], batch_size: int = 64) -> torch.Tensor:
        """
        Extracts mean-pooled contextual embeddings for a list of sentences in mini-batches
        to prevent Out-Of-Memory errors on long documents.
        """
        if not sents:
            return torch.zeros((0, self.hidden_dim))

        all_embeddings = []

        # Process in mini-batches of 64 sentences
        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i : i + batch_size]

            encoded = self.tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**encoded)
                token_embeddings = outputs.last_hidden_state  # [Batch, SeqLen, 768]
                
                mask = encoded['attention_mask'].unsqueeze(-1).float()
                sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                
                batch_embeddings = (sum_embeddings / sum_mask).cpu()
                all_embeddings.append(batch_embeddings)

        return torch.cat(all_embeddings, dim=0)