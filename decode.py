import math
from typing import List

import torch
from sherpa import RnntConformerModel, greedy_search, modified_beam_search
from torch.nn.utils.rnn import pad_sequence

LOG_EPS = math.log(1e-10)


@torch.no_grad()
def run_model_and_do_greedy_search(
    model: RnntConformerModel,
    features: List[torch.Tensor],
) -> List[List[int]]:
    """Run RNN-T model with the given features and use greedy search
    to decode the output of the model.

    Args:
      model:
        The RNN-T model.
      features:
        A list of 2-D tensors. Each entry is of shape
        (num_frames, feature_dim).
    Returns:
      Return a list-of-list containing the decoding token IDs.
    """
    features_length = torch.tensor(
        [f.size(0) for f in features],
        dtype=torch.int64,
    )
    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=LOG_EPS,
    )

    device = model.device
    features = features.to(device)
    features_length = features_length.to(device)

    encoder_out, encoder_out_length = model.encoder(
        features=features,
        features_length=features_length,
    )

    hyp_tokens = greedy_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_length=encoder_out_length.cpu(),
    )
    return hyp_tokens


@torch.no_grad()
def run_model_and_do_modified_beam_search(
    model: RnntConformerModel,
    features: List[torch.Tensor],
    num_active_paths: int,
) -> List[List[int]]:
    """Run RNN-T model with the given features and use greedy search
    to decode the output of the model.

    Args:
      model:
        The RNN-T model.
      features:
        A list of 2-D tensors. Each entry is of shape
        (num_frames, feature_dim).
      num_active_paths:
        Used only when decoding_method is modified_beam_search.
        It specifies number of active paths for each utterance. Due to
        merging paths with identical token sequences, the actual number
        may be less than "num_active_paths".
    Returns:
      Return a list-of-list containing the decoding token IDs.
    """
    features_length = torch.tensor(
        [f.size(0) for f in features],
        dtype=torch.int64,
    )
    features = pad_sequence(
        features,
        batch_first=True,
        padding_value=LOG_EPS,
    )

    device = model.device
    features = features.to(device)
    features_length = features_length.to(device)

    encoder_out, encoder_out_length = model.encoder(
        features=features,
        features_length=features_length,
    )

    hyp_tokens = modified_beam_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_length=encoder_out_length.cpu(),
        num_active_paths=num_active_paths,
    )
    return hyp_tokens
