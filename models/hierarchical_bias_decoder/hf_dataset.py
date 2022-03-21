# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import torch.nn as nn
import math
from fairseq.data import FairseqDataset, data_utils

from anytree import Node, Walker

logger = logging.getLogger(__name__)


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


def collate(
    samples,
    pad_idx,
    eos_idx,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
    hier2index=None
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, pad_idx, move_eos_to_beginning=False, pad_to_length=None, pad_to_multiple=1):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=False,
        pad_idx=pad_idx,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        pad_to_multiple=pad_to_multiple
    )
    attention_mask = (src_tokens != pad_idx).long()
    global_attention_mask = torch.zeros_like(src_tokens, dtype=torch.long)
    global_attention_mask[:, 0] = 1
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    max_input_length = src_tokens.size(1)

    hier_bias = torch.full((len(samples), max_input_length, max_input_length), 0, dtype=torch.long)
    w = Walker()
    for bid, sample in enumerate(samples):
        pad_size = max_input_length - len(sample["source"])
        chunk_size = sample["chunk_sizes"]
        padded_chunk_size = chunk_size[:-1] + [chunk_size[-1] + pad_size]
        parent_ids = sample["parent_ids"]

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

                hier_bias[bid, s1_start:s1_end, s2_start:s2_end] = hier2index[(path_length, level_diff)]
                hier_bias[bid, s2_start:s2_end, s1_start:s1_end] = hier2index[(-path_length, -level_diff)]

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=False,
            pad_idx=pad_idx,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            "target",
            left_pad=False,
            pad_idx=pad_idx,
            move_eos_to_beginning=True,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "input_ids": src_tokens,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "cross_attn_hierarchical_bias": hier_bias,
            "decoder_input_ids": prev_output_tokens,
        },
        "target": target,
    }

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        raw_dataset,
        src_sizes,
        tgt_sizes,
        pad=1,
        eos=2,
        shuffle=True,
        input_feeding=True,
        pad_to_multiple=1,
    ):
        self.raw_dataset = raw_dataset
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.pad = pad
        self.eos = eos
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        hier2index = {}
        for possible_path_length in range(20):
            if possible_path_length % 2 == 0:
                for possible_level_diff in range(0, possible_path_length + 1, 2):
                    hier2index[(possible_path_length, possible_level_diff)] = len(hier2index)
                    if possible_path_length != 0:
                        hier2index[(-possible_path_length, possible_level_diff)] = len(hier2index)
                    if possible_level_diff != 0:
                        hier2index[(possible_path_length, -possible_level_diff)] = len(hier2index)
                        if possible_path_length != 0:
                            hier2index[(-possible_path_length, -possible_level_diff)] = len(hier2index)
            else:
                for possible_level_diff in range(1, possible_path_length + 1, 2):
                    hier2index[(possible_path_length, possible_level_diff)] = len(hier2index)
                    hier2index[(-possible_path_length, possible_level_diff)] = len(hier2index)
                    hier2index[(possible_path_length, -possible_level_diff)] = len(hier2index)
                    hier2index[(-possible_path_length, -possible_level_diff)] = len(hier2index)
        self.hier2index = hier2index

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        item = self.raw_dataset[int(index)]
        example = {
            "id": index,
            "source": torch.tensor(item['tokenized_inputs'], dtype=torch.long),
            "target": torch.tensor(item['labels']),
            "chunk_sizes": item['chunk_sizes'],
            "parent_ids": item['parent_ids'],
        }
        return example

    def __len__(self):
        return len(self.raw_dataset)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.pad,
            eos_idx=self.eos,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
            hier2index=self.hier2index
        )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
