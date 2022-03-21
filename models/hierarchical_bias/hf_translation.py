from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from .hf_dataset import LanguagePairDataset

import datasets
from transformers import AutoTokenizer
from datasets import load_dataset


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


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


def load_langpair_dataset(
    data_path,
    data_loading_file,
    data_name,
    split,
    tokenizer,
    max_source_positions,
    max_target_positions,
    shuffle=True,
    pad_to_multiple=1,
    overwrite_cache=False,
):
    if split == 'train':
        split = datasets.Split.TRAIN
    elif split == 'valid':
        split = datasets.Split.VALIDATION
    else:
        split = datasets.Split.TEST

    raw_dataset = load_dataset(data_loading_file, name=data_name, data_dir=data_path, split=split)
    column_names = raw_dataset.column_names

    def preprocess_function(examples):
        document_paragraphs = examples["document_paragraphs"]
        document_section_ends = examples["section_paragraph_ends"]
        document_section_depths = examples["section_depths"]
        document_section_titles = examples["section_titles"]
        document_section_parent_ids = examples["section_parent_ids"]
        summary = examples["summary"]

        tokenized_document_paragraphs = [tokenizer(document_paragraph, add_special_tokens=False)["input_ids"] for document_paragraph in document_paragraphs]
        tokenized_section_titles = [tokenizer(document_section_title, add_special_tokens=False)["input_ids"] for document_section_title in document_section_titles]

        tokenized_document = []
        section_chunk_sizes = []
        section_parent_ids = []
        section_depths = []
        for tokenized_document_paragraph, document_section_end, document_section_depth, tokenized_section_title, document_section_parent_id in zip(tokenized_document_paragraphs, document_section_ends, document_section_depths, tokenized_section_titles, document_section_parent_ids):
            assert len(document_section_end) == len(document_section_depth) == len(tokenized_section_title) == len(document_section_parent_id)
            previous_end = 0
            accumulate_length = 0
            accumulate_section = []
            for section_end, section_depth, section_parent_id, section_title in zip(document_section_end, document_section_depth, document_section_parent_id, tokenized_section_title):
                accumulate_length += 1  # for section special token
                accumulate_paragraph = [section_title]
                accumulate_length += len(section_title)
                if accumulate_length > max_source_positions - 2:
                    break
                for section_paragraph in tokenized_document_paragraph[previous_end:section_end]:
                    accumulate_paragraph.append(section_paragraph[:max_source_positions - 2 - accumulate_length])
                    accumulate_length += len(section_paragraph) + 1
                    if accumulate_length > max_source_positions - 2:
                        break
                if accumulate_length > max_source_positions - 2:
                    break
                accumulate_section.append((accumulate_paragraph[0] + [tok for paragraph in accumulate_paragraph[1:] for tok in [48900] + paragraph], section_depth, section_parent_id))
                previous_end = section_end

            document = []
            accumulate_parent_ids = []
            chunk_size = []
            previous_token_end = 0
            depths = []
            for section, depth, parent_id in accumulate_section:
                accumulate_parent_ids.append(parent_id)
                depths.append(depth)
                document.extend([DEPTH_SPECIAL_TOKENS.get(depth, 49216)] + section)
                chunk_size.append(len(document) - previous_token_end)
                previous_token_end = len(document)

            document = [0] + document + [2]
            chunk_size = [1] + chunk_size + [1]
            depths = [0] + depths + [0]
            accumulate_parent_ids = [id if id == -1 else id + 1 for id in accumulate_parent_ids]
            accumulate_parent_ids = [-1] + accumulate_parent_ids + [-1]
            assert len(chunk_size) == len(accumulate_parent_ids)

            tokenized_document.append(document)
            section_chunk_sizes.append(chunk_size)
            section_parent_ids.append(accumulate_parent_ids)
            section_depths.append(depths)

        with tokenizer.as_target_tokenizer():
            tokenized_summary = tokenizer(summary, truncation=True, max_length=max_target_positions)["input_ids"]

        examples["tokenized_inputs"] = tokenized_document
        examples["labels"] = tokenized_summary
        examples["chunk_sizes"] = section_chunk_sizes
        examples["parent_ids"] = section_parent_ids
        examples["depths"] = section_depths
        return examples

    def preprocess_function_qs_fq(examples):
        document_paragraphs = examples["document_paragraphs"]
        document_section_ends = examples["section_paragraph_ends"]
        document_section_depths = examples["section_depths"]
        document_section_titles = examples["section_titles"]
        document_section_parent_ids = examples["section_parent_ids"]
        first_questions = examples["first_question"]
        first_summaries = examples["first_summary"]
        document_summary_paragraphs = examples["summary_paragraphs"]
        document_summary_depths = examples["summary_depths"]

        tokenized_document_paragraphs = [tokenizer(document_paragraph, add_special_tokens=False)["input_ids"] for
                                         document_paragraph in document_paragraphs]
        tokenized_section_titles = [tokenizer(document_section_title, add_special_tokens=False)["input_ids"] for
                                    document_section_title in document_section_titles]

        tokenized_first_questions = tokenizer(first_questions, add_special_tokens=False)["input_ids"]

        tokenized_document = []
        section_chunk_sizes = []
        section_parent_ids = []
        for tokenized_document_paragraph, document_section_end, document_section_depth, tokenized_section_title, document_section_parent_id, \
                tokenized_first_question in zip(
                tokenized_document_paragraphs, document_section_ends, document_section_depths, tokenized_section_titles,
                document_section_parent_ids, tokenized_first_questions):
            assert len(document_section_end) == len(document_section_depth) == len(tokenized_section_title) == len(
                document_section_parent_id)
            previous_end = 0
            accumulate_length = 1 + len(tokenized_first_question)
            accumulate_section = []
            for section_end, section_depth, section_parent_id, section_title in zip(document_section_end,
                                                                                    document_section_depth,
                                                                                    document_section_parent_id,
                                                                                    tokenized_section_title):
                accumulate_length += 1  # for section special token
                accumulate_paragraph = [section_title]
                accumulate_length += len(section_title)
                if accumulate_length > max_source_positions - 2:
                    break
                for section_paragraph in tokenized_document_paragraph[previous_end:section_end]:
                    accumulate_paragraph.append(section_paragraph[:max_source_positions - 2 - accumulate_length])
                    accumulate_length += len(section_paragraph) + 1
                    if accumulate_length > max_source_positions - 2:
                        break
                if accumulate_length > max_source_positions - 2:
                    break
                accumulate_section.append((accumulate_paragraph[0] + [tok for paragraph in accumulate_paragraph[1:] for
                                                                      tok in [48900] + paragraph], section_depth,
                                           section_parent_id))
                previous_end = section_end

            document = []
            accumulate_parent_ids = []
            chunk_size = []
            previous_token_end = 0
            for section, depth, parent_id in accumulate_section:
                accumulate_parent_ids.append(parent_id)
                document.extend([DEPTH_SPECIAL_TOKENS.get(depth, 49216)] + section)
                chunk_size.append(len(document) - previous_token_end)
                previous_token_end = len(document)

            document = [0] + tokenized_first_question + [0] + document + [2]
            chunk_size = [1 + len(tokenized_first_question)] + [1] + chunk_size + [1]
            accumulate_parent_ids = [id if id == -1 else id + 2 for id in accumulate_parent_ids]
            accumulate_parent_ids = [-1] + [-1] + accumulate_parent_ids + [-1]
            assert len(chunk_size) == len(accumulate_parent_ids)

            tokenized_document.append(document)
            section_chunk_sizes.append(chunk_size)
            section_parent_ids.append(accumulate_parent_ids)

        with tokenizer.as_target_tokenizer():
            tokenized_document_summary_paragraphs = [
                tokenizer(document_paragraph, add_special_tokens=False)["input_ids"] if document_paragraph else [] for document_paragraph in
                document_summary_paragraphs]

            tokenized_first_summaries = tokenizer(first_summaries, add_special_tokens=False)["input_ids"]

            tokenized_summary = []
            for tokenized_document_summary_paragraph, document_summary_depth, first_summary in zip(
                    tokenized_document_summary_paragraphs, document_summary_depths, tokenized_first_summaries):
                current_depth = 0
                summary = []
                for summary_paragraph, summary_depth in zip(tokenized_document_summary_paragraph,
                                                            document_summary_depth):
                    if summary_depth > current_depth:
                        summary = summary + [ACTION_SPECIAL_TOKENS["DOWN"]] * (summary_depth - current_depth)
                        summary = summary + summary_paragraph
                    elif summary_depth == current_depth:
                        summary = summary + [ACTION_SPECIAL_TOKENS["HOLD"]]
                        summary = summary + summary_paragraph
                    else:
                        summary = summary + [ACTION_SPECIAL_TOKENS["UP"]] * (current_depth - summary_depth)
                        summary = summary + summary_paragraph
                summary = [0] + first_summary + summary[:max_target_positions - len(first_summary) - 2] + [2]
                tokenized_summary.append(summary)

        examples["tokenized_inputs"] = tokenized_document
        examples["labels"] = tokenized_summary
        examples["chunk_sizes"] = section_chunk_sizes
        examples["parent_ids"] = section_parent_ids

        return examples

    def preprocess_function_qs_qg(examples):
        document_paragraphs = examples["document_paragraphs"]
        document_section_ends = examples["section_paragraph_ends"]
        document_section_depths = examples["section_depths"]
        document_section_titles = examples["section_titles"]
        document_section_parent_ids = examples["section_parent_ids"]
        first_questions = examples["first_question"]
        first_summaries = examples["first_summary"]
        document_summary_paragraphs = examples["summary_paragraphs"]

        tokenized_document_paragraphs = [tokenizer(document_paragraph, add_special_tokens=False)["input_ids"] for
                                         document_paragraph in document_paragraphs]
        tokenized_section_titles = [tokenizer(document_section_title, add_special_tokens=False)["input_ids"] for
                                    document_section_title in document_section_titles]

        tokenized_first_questions = tokenizer(first_questions, add_special_tokens=False)["input_ids"]
        tokenized_first_summaries = tokenizer(first_summaries, add_special_tokens=False)["input_ids"]

        tokenized_document = []
        section_chunk_sizes = []
        section_parent_ids = []
        for tokenized_document_paragraph, document_section_end, document_section_depth, tokenized_section_title, document_section_parent_id, \
                tokenized_first_question, tokenized_first_summary in zip(
                tokenized_document_paragraphs, document_section_ends, document_section_depths, tokenized_section_titles,
                document_section_parent_ids, tokenized_first_questions, tokenized_first_summaries):
            assert len(document_section_end) == len(document_section_depth) == len(tokenized_section_title) == len(
                document_section_parent_id)
            previous_end = 0
            accumulate_length = 1 + len(tokenized_first_question) + len(tokenized_first_summary)
            accumulate_section = []
            for section_end, section_depth, section_parent_id, section_title in zip(document_section_end,
                                                                                    document_section_depth,
                                                                                    document_section_parent_id,
                                                                                    tokenized_section_title):
                accumulate_length += 1  # for section special token
                accumulate_paragraph = [section_title]
                accumulate_length += len(section_title)
                if accumulate_length > max_source_positions - 2:
                    break
                for section_paragraph in tokenized_document_paragraph[previous_end:section_end]:
                    accumulate_paragraph.append(section_paragraph[:max_source_positions - 2 - accumulate_length])
                    accumulate_length += len(section_paragraph) + 1
                    if accumulate_length > max_source_positions - 2:
                        break
                if accumulate_length > max_source_positions - 2:
                    break
                accumulate_section.append((accumulate_paragraph[0] + [tok for paragraph in accumulate_paragraph[1:] for
                                                                      tok in [48900] + paragraph], section_depth,
                                           section_parent_id))
                previous_end = section_end

            document = []
            accumulate_parent_ids = []
            chunk_size = []
            previous_token_end = 0
            for section, depth, parent_id in accumulate_section:
                accumulate_parent_ids.append(parent_id)
                document.extend([DEPTH_SPECIAL_TOKENS.get(depth, 49216)] + section)
                chunk_size.append(len(document) - previous_token_end)
                previous_token_end = len(document)

            document = [0] + tokenized_first_question + tokenized_first_summary + [0] + document + [2]
            chunk_size = [1 + len(tokenized_first_question) + len(tokenized_first_summary)] + [1] + chunk_size + [1]
            accumulate_parent_ids = [id if id == -1 else id + 2 for id in accumulate_parent_ids]
            accumulate_parent_ids = [-1] + [-1] + accumulate_parent_ids + [-1]
            assert len(chunk_size) == len(accumulate_parent_ids)

            tokenized_document.append(document)
            section_chunk_sizes.append(chunk_size)
            section_parent_ids.append(accumulate_parent_ids)

        with tokenizer.as_target_tokenizer():
            tokenized_document_summary_paragraphs = [
                tokenizer(document_paragraph, add_special_tokens=False)["input_ids"] if document_paragraph else [] for document_paragraph in
                document_summary_paragraphs]

            tokenized_summary = []
            for tokenized_document_summary_paragraph in tokenized_document_summary_paragraphs:
                summary = []
                for summary_paragraph in tokenized_document_summary_paragraph:
                    if summary:
                        summary = summary + [ACTION_SPECIAL_TOKENS["HOLD"]]
                    summary = summary + summary_paragraph
                summary = summary[:max_target_positions - 2] + [2]
                tokenized_summary.append(summary)

        examples["tokenized_inputs"] = tokenized_document
        examples["labels"] = tokenized_summary
        examples["chunk_sizes"] = section_chunk_sizes
        examples["parent_ids"] = section_parent_ids

        return examples

    if data_name == 'qs_hierarchy_fq':
        raw_dataset = raw_dataset.map(
            preprocess_function_qs_fq,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {split} dataset",
        )
    elif data_name == 'qs_hierarchy_qg':
        raw_dataset = raw_dataset.map(
            preprocess_function_qs_qg,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {split} dataset",
        )
    elif data_name == 'gov_report' or data_name == 'wiki_bio_sum':
        raw_dataset = raw_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=column_names,
            load_from_cache_file=not overwrite_cache,
            desc=f"Running tokenizer on {split} dataset",
        )
    else:
        raise ValueError(f"Unknown data name: {data_name}")

    src_sizes = [len(example["tokenized_inputs"]) for example in raw_dataset]
    tgt_sizes = [len(example["labels"]) for example in raw_dataset]

    return LanguagePairDataset(
        raw_dataset,
        src_sizes,
        tgt_sizes,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class HuggingFaceTranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    data_loading_file: Optional[str] = field(
        default=None
    )
    data_name: str = field(
        default='gov_report'
    )
    max_source_positions: int = field(
        default=16384, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")
    tokenizer_name: str = field(
        default='allenai/led-large-16384'
    )


@register_task("hf_translation", dataclass=HuggingFaceTranslationConfig)
class HuggingfaceTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.
    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language
    .. note::
        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: HuggingFaceTranslationConfig

    def __init__(self, cfg: HuggingFaceTranslationConfig, tokenizer):
        super().__init__(cfg)
        self.tokenizer = tokenizer

    @classmethod
    def setup_task(cls, cfg: HuggingFaceTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)

        logger.info("tokenizer: {}".format(cfg.tokenizer_name))

        return cls(cfg, tokenizer)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        self.datasets[split] = load_langpair_dataset(
            self.cfg.data,
            self.cfg.data_loading_file,
            self.cfg.data_name,
            split,
            self.tokenizer,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return None

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return None
