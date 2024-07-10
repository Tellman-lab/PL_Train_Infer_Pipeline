import torch
import torch.nn as nn


def feature_encoding(feature_dim, n_features, n_heads, n_layers, batch_size):
    # Build model.
    encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim,
                                               nhead=n_heads,
                                               dim_feedforward=feature_dim,
                                               batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    # Create sample input.
    x = torch.zeros([batch_size, n_features, feature_dim])

    # Pass sample input into model
    y = transformer_encoder(x)

    # Get output shape.
    print(y.shape)  # [batch_size, n_features, feature_dim]


def special_token(feature_dim, n_features, n_heads, n_layers, batch_size):
    # Build model.
    encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim,
                                               nhead=n_heads,
                                               dim_feedforward=feature_dim,
                                               batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    # Create sample input.
    x = torch.zeros([batch_size, n_features, feature_dim])

    # Create special token.
    special_token = nn.Parameter(torch.rand(size=[1, n_features, feature_dim]))
    special_token = special_token.repeat((batch_size, 1, 1))

    # Combine special token with other features.
    x = torch.concat([x, special_token], dim=1)

    # Pass sample input into model
    y = transformer_encoder(x)

    # Get refined special token features.
    output = y[:, -1]
    print(output.shape)


if __name__ == '__main__':
    feature_dim = 1024  # F
    n_features = 6  # S
    n_heads = 4
    n_layers = 2
    batch_size = 2  # B

    feature_encoding(feature_dim, n_features, n_heads, n_layers, batch_size)
    special_token(feature_dim, n_features, n_heads, n_layers, batch_size)
