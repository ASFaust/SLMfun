import torch
import torch.nn as nn
import torch.nn.functional as F


class KVStore(nn.Module):
    def __init__(
            self,
            storage_size,
            key_dim,
            value_dim,
            top_k,
            device
    ):
        super(KVStore, self).__init__()
        self.storage_size = storage_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.top_k = top_k
        self.device = device
        self.storage = nn.Parameter(torch.randn((storage_size, key_dim + value_dim), device=device))

    def forward(self, x):
        # x shape: (batch_size, key_dim)

        # Split storage into keys and values
        keys = self.storage[:, :self.key_dim]  # (storage_size, key_dim)
        values = self.storage[:, self.key_dim:]  # (storage_size, value_dim)

        # Normalize keys and the query x to project onto the hypersphere
        keys = F.normalize(keys, p=2, dim=1)  # L2 normalization on each key vector
        query = F.normalize(x, p=2, dim=1)  # L2 normalization on query

        # Compute similarity between each query and all keys
        similarities = torch.matmul(query, keys.T)  # (batch_size, storage_size)

        # Retrieve the indices of the top-k most similar keys for each query in the batch
        topk_similarities, topk_indices = torch.topk(similarities, self.top_k, dim=1)

        # Gather top-k values based on indices for each query
        # Expanding dimensions to match batch processing
        topk_values = torch.gather(values.expand(query.size(0), -1, -1), 1,
                                   topk_indices.unsqueeze(-1).expand(-1, -1, self.value_dim))

        # Compute softmax weights over top-k similarities for focused retrieval
        topk_weights = F.softmax(topk_similarities, dim=1).unsqueeze(-1)  # (batch_size, top_k, 1)

        # Weighted sum of top-k values for each query
        output = (topk_weights * topk_values).sum(dim=1)  # (batch_size, value_dim)

        return output
