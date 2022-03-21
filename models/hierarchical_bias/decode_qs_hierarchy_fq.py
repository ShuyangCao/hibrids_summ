import argparse
import os
import torch

from tqdm import tqdm

import torch.utils.data
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import OrderedDict

from seq2seq_collator import DataCollatorForLinearTitle, DEPTH_SPECIAL_TOKENS, ACTION_SPECIAL_TOKENS
from modeling_led import LEDForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join(os.getenv('EXPDIR'), 'data'))
    parser.add_argument('--model_dir')
    parser.add_argument('--output_dir', default=os.path.join(os.getenv('EXPDIR'), 'decode_outputs/qs_hierarchy_fq/hierarchical_bias'))
    parser.add_argument('--max_length', type=int, default=5120)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_id', type=int, default=0)
    args = parser.parse_args()

    test_dataset = load_dataset('../data_loading_script/summary_generation_data.py', name='qs_hierarchy_fq',
                                data_dir=args.data_dir, split='test')
    column_names = test_dataset.column_names

    tokenizer = AutoTokenizer.from_pretrained('allenai/led-large-16384', use_fast=True)
    model = LEDForConditionalGeneration.from_pretrained('allenai/led-large-16384', use_cache=True)

    ckpt = torch.load(os.path.join(args.model_dir, 'checkpoint_best.pt'))
    new_state_dict = OrderedDict()
    for k, v in ckpt['model'].items():
        if k.startswith('hf_model'):
            new_state_dict[k[len('hf_model.'):]] = v

    model.load_state_dict(new_state_dict)

    model.cuda()
    model.eval()

    def preprocess_function(examples):
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
        for tokenized_document_paragraph, document_section_end, document_section_depth, tokenized_section_title, document_section_parent_id, tokenized_first_question in zip(
                tokenized_document_paragraphs, document_section_ends, document_section_depths, tokenized_section_titles,
                document_section_parent_ids, tokenized_first_questions):
            assert len(document_section_end) == len(document_section_depth) == len(tokenized_section_title) == len(
                document_section_parent_id)
            previous_end = 0
            accumulate_length = 1 + len(tokenized_first_questions)
            accumulate_section = []
            for section_end, section_depth, section_parent_id, section_title in zip(document_section_end,
                                                                                    document_section_depth,
                                                                                    document_section_parent_id,
                                                                                    tokenized_section_title):
                accumulate_length += 1  # for section special token
                accumulate_paragraph = [section_title]
                accumulate_length += len(section_title)
                if accumulate_length > args.max_length - 2:
                    break
                for section_paragraph in tokenized_document_paragraph[previous_end:section_end]:
                    accumulate_paragraph.append(section_paragraph[:args.max_length - 2 - accumulate_length])
                    accumulate_length += len(section_paragraph) + 1
                    if accumulate_length > args.max_length - 2:
                        break
                if accumulate_length > args.max_length - 2:
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
                tokenizer(document_paragraph, add_special_tokens=False)["input_ids"] if document_paragraph else []
                for document_paragraph in
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
                summary = [0] + first_summary + summary[:1024 - len(first_summary) - 2] + [2]
                tokenized_summary.append(summary)

        examples["tokenized_inputs"] = tokenized_document
        examples["labels"] = tokenized_summary
        examples["chunk_sizes"] = section_chunk_sizes
        examples["parent_ids"] = section_parent_ids

        return examples

    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on prediction dataset",
    )

    if args.num_shards > 1:
        test_dataset = test_dataset.shard(num_shards=args.num_shards, index=args.shard_id, contiguous=True)

    label_pad_token_id = -100
    data_collator = DataCollatorForLinearTitle(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=1024,
        max_length=args.max_length
    )

    with torch.no_grad():
        predictions = []
        targets = []
        for sample in tqdm(torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=data_collator, shuffle=False)):
            labels = sample.pop("labels")
            decoder_input_ids = sample.pop("decoder_input_ids")

            for k in sample:
                if isinstance(sample[k], torch.Tensor):
                    sample[k] = sample[k].to('cuda')

            predicted_ids = model.generate(**sample, max_length=1024, num_beams=4, early_stopping=True, no_repeat_ngram_size=5)

            prediction = tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            labels = labels.masked_fill(labels == -100, tokenizer.pad_token_id)

            target = tokenizer.batch_decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            predictions.extend(prediction)
            targets.extend(target)

        os.makedirs(args.output_dir, exist_ok=True)

        output_prediction_file = os.path.join(args.output_dir, "generated_predictions.txt")
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions) + '\n')

        output_target_file = os.path.join(args.output_dir, "targets.txt")
        with open(output_target_file, "w") as writer:
            writer.write("\n".join(targets) + '\n')




if __name__ == '__main__':
    main()