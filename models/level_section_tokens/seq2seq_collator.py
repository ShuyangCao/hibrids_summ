from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy, BatchEncoding


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
            "labels": labels
        }

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            all_features["decoder_input_ids"] = decoder_input_ids

        return all_features
