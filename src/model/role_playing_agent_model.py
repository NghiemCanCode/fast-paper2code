"""
Role-playing Agent Model implementation.

This module contains the implementation of the Role-playing Agent Model, which public at HUFLIT JOURNAL OF SCIENCE.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertModel

from .base_model import BaseModel


def _freeze_layers(models: List[nn.Module], layers_to_unfreeze: List[int]) -> None:
    """
    Freeze all layers in the models except for the specified layers to unfreeze.

    Args:
        models: List of models to apply freezing to
        layers_to_unfreeze: List of layer indices to keep unfrozen (requires_grad=True)
    """
    for model, layer_idx in zip(models, layers_to_unfreeze):
        for name, param in model.named_parameters():
            # By default, freeze all parameters
            param.requires_grad = True
            # Unfreeze only the specified layer
            # if name.startswith(f"encoder.layer.{layer_idx}"):
            #     param.requires_grad = True


class MaskAttentionLayer(nn.Module):
    """
    Attention layer with mask support for focusing on relevant features.

    This layer applies the attention mechanism to input features, optionally
    considering a mask to ignore certain positions.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Initialize the mask attention layer.

        Args:
            input_dim: Dimension of the input features
        """
        super(MaskAttentionLayer, self).__init__()
        self._attention_layer = nn.Linear(input_dim, 1)

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the attention mechanism to the inputs.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask tensor of shape [batch_size, seq_len]

        Returns:
            Tuple containing:
                - outputs: Weighted sum of input features
                - scores: Attention scores
        """
        scores = self._attention_layer(inputs).view(-1, inputs.size(1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = torch.softmax(scores, dim=1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)

        return outputs, scores


class MLP(nn.Module):
    """
    Multi-Layer Perceptron implementation.

    A flexible MLP that can be configured with different hidden dimensions,
    dropout rates, and an optional output layer.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            dropout_rate: float,
            output_layer: bool = True
    ) -> None:
        """
        Initialize the MLP.

        Args:
            input_dim: Dimension of the input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            output_layer: Whether to add a final linear layer with output dim=1
        """
        super(MLP, self).__init__()
        self._layers = nn.ModuleList()

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self._layers.append(nn.Linear(current_dim, hidden_dim))
            self._layers.append(nn.ReLU())
            self._layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        if output_layer:
            self._layers.append(nn.Linear(current_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all layers
        """
        for layer in self._layers:
            x = layer(x)
        return x


class RolePlayingAgentModel(BaseModel):
    """
    Model for fact-checking with multi-agent.

    This model processes news and explanations to classify their authenticity.
    It uses BERT embedding and attention mechanisms to aggregate features.
    """

    def __init__(self, config: Dict[str, Union[str, int, float]]) -> None:
        """
        initialize the RolePlayingAgentModel.
        :param config: dict of config parameters.
        the accepted keys in config are:
        - embedding_model: str, the embedding model name.
        - embedding_dim_attention: int, the dimension of the attention layer.
        - mlp_hidden_dims: list of int, the hidden dimensions of the MLP.
        :return: (None)
        """
        super(RolePlayingAgentModel, self).__init__()
        self._config = config

        # Initialize BERT models for embeddings
        self._news_embedding = BertModel.from_pretrained(config['embedding_model']).requires_grad_(False)
        self._good_explanation_embedding = BertModel.from_pretrained(config['embedding_model']).requires_grad_(False)
        self._bad_explanation_embedding = BertModel.from_pretrained(config['embedding_model']).requires_grad_(False)
        self._ugly_explanation_embedding = BertModel.from_pretrained(config['embedding_model']).requires_grad_(False)

        for i in range(10, 12):
            for name, param in self._news_embedding.named_parameters():
                if name.startswith(f"encoder.layer.{i}"):
                    param.requires_grad = True

        for i in range(10, 12):
            for name, param in self._good_explanation_embedding.named_parameters():
                if name.startswith(f"encoder.layer.{i}"):
                    param.requires_grad = True

        for i in range(10, 12):
            for name, param in self._bad_explanation_embedding.named_parameters():
                if name.startswith(f"encoder.layer.{i}"):
                    param.requires_grad = True

        for i in range(11, 12):
            for name, param in self._ugly_explanation_embedding.named_parameters():
                if name.startswith(f"encoder.layer.{i}"):
                    param.requires_grad = True

        # Initialize attention layers
        embedding_dim = config['embedding_dim_attention']
        self._feature_aggregator = MaskAttentionLayer(embedding_dim)
        self._news_attention = MaskAttentionLayer(embedding_dim)
        self._good_attention = MaskAttentionLayer(embedding_dim)
        self._bad_attention = MaskAttentionLayer(embedding_dim)
        self._ugly_attention = MaskAttentionLayer(embedding_dim)

        # Initialize classifier
        self._classifier = MLP(
            input_dim=embedding_dim,
            hidden_dims=[384],
            dropout_rate=0.2
        )

    def name(self) -> str:
        """
        :return: Name of the model.
        """
        return "role_playing_agent_model"

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.

        :param kwargs: Keyword arguments contain:
            - news_token_ids: News text tensor
            - news_mask: Attention mask for news
            - good_explain_token_ids: Good explanations tensor
            - good_explain_mask: Attention mask for good explanations
            - bad_explain_token_ids: Bad explanations tensor
            - bad_explain_mask: Attention mask for bad explanations
            - ugly_explain_token_ids: Ugly explanations tensor
            - ugly_explain_mask: Attention mask for ugly explanations
            - is_training: Whether the model is training.

        :return:
            Classification prediction tensor
        """
        # Extract inputs
        news = kwargs['news_token_ids']
        news_mask = kwargs['news_mask']
        good_explanations = kwargs['good_explain_token_ids']
        good_explanations_mask = kwargs['good_explain_mask']
        bad_explanations = kwargs['bad_explain_token_ids']
        bad_explanations_mask = kwargs['bad_explain_mask']
        ugly_explanations = kwargs['ugly_explain_token_ids']
        ugly_explanations_mask = kwargs['ugly_explain_mask']

        # Get embeddings
        news_features = self._news_embedding(news, attention_mask=news_mask)[0]
        good_features = self._good_explanation_embedding(good_explanations, attention_mask=good_explanations_mask)[0]
        bad_features = self._bad_explanation_embedding(bad_explanations, attention_mask=bad_explanations_mask)[0]
        ugly_features = self._ugly_explanation_embedding(ugly_explanations, attention_mask=ugly_explanations_mask)[0]

        news_features, _ = self._news_attention(news_features, mask=news_mask)
        good_features, _ = self._good_attention(good_features, mask=good_explanations_mask)
        bad_features, _ = self._bad_attention(bad_features, mask=bad_explanations_mask)
        ugly_features, _ = self._ugly_attention(ugly_features, mask=ugly_explanations_mask)

        news_features += ugly_features

        # Aggregate features
        all_features = torch.cat([
            news_features.unsqueeze(1),
            good_features.unsqueeze(1),
            bad_features.unsqueeze(1)
        ], dim=1)

        final_feature, _ = self._feature_aggregator(all_features)

        # Classification
        logits = self._classifier(final_feature)
        predictions = torch.sigmoid(logits.squeeze(1))

        return predictions