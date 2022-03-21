from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from anytree import Node, Walker

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy, BatchEncoding
import torch.nn as nn


DEPTH_SPECIAL_TOKENS = {
    -1: 48900,
    0: 48613,
    1: 48983,
    2: 48936,
    3: 48712,
    4: 49130,
    5: 49216
}

ACTION_SPECIAL_TOKENS = {
    "UP": 49908,
    "HOLD": 49859,
    "DOWN": 49452
}


def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
    """pads rows and then flips rows and columns"""
    hidden_states_padded = nn.functional.pad(
        hidden_states_padded, padding
    )  # padding value is not important because it will be overwritten
    hidden_states_padded = hidden_states_padded.view(
        *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
    )
    return hidden_states_padded


def _chunk_bias(biases, window_overlap):
    # biases: bsz x seq_len x seq_len
    # -> bsz x seq_len x num_chunk x window_size
    batch_size, seq_len, _ = biases.size()

    # chunks_count = seq_len // window_overlap - 1

    biases = biases.view(
        biases.size(0),
        biases.size(1),
        biases.size(2) // (window_overlap * 2),
        window_overlap * 2
    )

    chunk_size = list(biases.size())
    chunk_size[2] = chunk_size[2] * 2 - 1

    chunk_stride = list(biases.stride())
    chunk_stride[2] = chunk_stride[2] // 2

    biases = biases.as_strided(size=chunk_size, stride=chunk_stride)

    _, _, num_chunks, _ = biases.size()
    biases = biases.view(
        batch_size,
        biases.size(1) // (window_overlap * 2),
        window_overlap * 2,
        num_chunks,
        window_overlap * 2,
    )

    chunk_size = list(biases.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(biases.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    biases = biases.as_strided(size=chunk_size, stride=chunk_stride)

    diagonal_biases = biases.new_empty(
        (batch_size, num_chunks, window_overlap * 2, window_overlap * 2)
    )
    for i in range(num_chunks):
        diagonal_biases[:, i] = biases[:, i, :, i]

    diagonal_biases = _pad_and_transpose_last_two_dims(
        diagonal_biases, padding=(0, 0, 0, 1)
    )

    diagonal_attention_scores = diagonal_biases.new_empty(
        (batch_size, num_chunks + 1, window_overlap, window_overlap * 2 + 1)
    )

    diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_biases[
                                                            :, :, :window_overlap, : window_overlap + 1
                                                            ]
    diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_biases[
                                                           :, -1, window_overlap:, : window_overlap + 1
                                                           ]
    # - copying the lower triangle
    diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_biases[
                                                           :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                           ]

    diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_biases[
                                                                          :, 0, : window_overlap - 1,
                                                                          1 - window_overlap:
                                                                          ]

    diagonal_attention_scores = diagonal_attention_scores.view(
        batch_size, 1, seq_len, 2 * window_overlap + 1
    ).transpose(2, 1)

    _mask_invalid_locations(diagonal_attention_scores, window_overlap, 0)

    diagonal_attention_scores = diagonal_attention_scores.transpose(2, 3)

    return diagonal_attention_scores


def _mask_invalid_locations(input_tensor, affected_seq_len, mask_value=-float("inf")):
    beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    beginning_mask = beginning_mask_2d[None, :, None, :]
    ending_mask = beginning_mask.flip(dims=(1, 3))
    beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    beginning_mask = beginning_mask.expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask == 1, mask_value)  # `== 1` converts to bool or uint8
    ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
    ending_mask = ending_mask.expand(ending_input.size())
    ending_input.masked_fill_(ending_mask == 1, mask_value)  # `== 1` converts to bool or uint8



HIER2INDEX = {}
for possible_path_length in range(20):
    if possible_path_length % 2 == 0:
        for possible_level_diff in range(0, possible_path_length + 1, 2):
            HIER2INDEX[(possible_path_length, possible_level_diff)] = len(HIER2INDEX)
            if possible_path_length != 0:
                HIER2INDEX[(-possible_path_length, possible_level_diff)] = len(HIER2INDEX)
            if possible_level_diff != 0:
                HIER2INDEX[(possible_path_length, -possible_level_diff)] = len(HIER2INDEX)
                if possible_path_length != 0:
                    HIER2INDEX[(-possible_path_length, -possible_level_diff)] = len(HIER2INDEX)
    else:
        for possible_level_diff in range(1, possible_path_length + 1, 2):
            HIER2INDEX[(possible_path_length, possible_level_diff)] = len(HIER2INDEX)
            HIER2INDEX[(-possible_path_length, possible_level_diff)] = len(HIER2INDEX)
            HIER2INDEX[(possible_path_length, -possible_level_diff)] = len(HIER2INDEX)
            HIER2INDEX[(-possible_path_length, -possible_level_diff)] = len(HIER2INDEX)


@dataclass
class DataCollatorForLinearTitle:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        labels = torch.tensor([feature['labels'] for feature in features], dtype=torch.long)

        input_ids = []
        attention_masks = []
        global_attention_masks = []
        for feature in features:
            input_id = feature["tokenized_inputs"]
            attention_mask = [1] * len(input_id)
            global_attention_mask = [0] * len(input_id)
            global_attention_mask[0] = 1
            input_id = input_id[:self.max_length][:-1] + [2]
            attention_mask = attention_mask[:self.max_length]
            global_attention_mask = global_attention_mask[:self.max_length]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            global_attention_masks.append(global_attention_mask)

        max_input_length = max(len(input_id) for input_id in input_ids)

        if self.pad_to_multiple_of is not None and (max_input_length % self.pad_to_multiple_of != 0):
            max_input_length = ((max_input_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        one_sided_window_size = self.model.config.attention_window[0] // 2
        section_chunk_sizes = [feature["chunk_sizes"] for feature in features]
        w = Walker()
        hier_bias = torch.full((len(features), max_input_length, max_input_length), 0, dtype=torch.long)
        for bid, feature in enumerate(features):
            pad_size = max_input_length - len(input_ids[bid])

            chunk_size = section_chunk_sizes[bid]
            padded_chunk_size = chunk_size[:-1] + [chunk_size[-1] + pad_size]
            parent_ids = feature["parent_ids"]

            root = Node("-1")
            nodes = []
            for cid, parent_id in enumerate(parent_ids):
                if parent_id == -1:
                    node = Node(str(cid), parent=root)
                else:
                    node = Node(str(cid), parent=nodes[parent_id])
                nodes.append(node)

            for s1 in range(len(parent_ids) - 1):
                s1_start = sum(padded_chunk_size[:s1])
                s1_end = s1_start + padded_chunk_size[s1]
                s1_node = nodes[s1]
                for s2 in range(s1 + 1, len(parent_ids)):
                    s2_start = sum(padded_chunk_size[:s2])
                    s2_end = s2_start + padded_chunk_size[s2]

                    assert s2_start >= s1_end, "might have error"

                    s2_node = nodes[s2]
                    upward, _, downward = w.walk(s1_node, s2_node)

                    level_diff = len(downward) - len(upward)
                    path_length = len(downward) + len(upward)

                    hier_bias[bid, s1_start:s1_end, s2_start:s2_end] = HIER2INDEX[(path_length, level_diff)]
                    hier_bias[bid, s2_start:s2_end, s1_start:s1_end] = HIER2INDEX[(-path_length, -level_diff)]

        # padding

        input_ids = [input_id + [self.tokenizer.pad_token_id] * (max_input_length - len(input_id)) for input_id in input_ids]
        attention_masks = [attention_mask + [0] * (max_input_length - len(attention_mask)) for attention_mask in attention_masks]
        global_attention_masks = [global_attention_mask + [0] * (max_input_length - len(global_attention_mask)) for global_attention_mask in global_attention_masks]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        global_attention_masks = torch.tensor(global_attention_masks, dtype=torch.long)

        all_features = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "global_attention_mask": global_attention_masks,
            "labels": labels,
            "cross_attn_hierarchical_bias": hier_bias
        }

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            all_features["decoder_input_ids"] = decoder_input_ids

        return all_features
