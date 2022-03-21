import argparse
import os
import numpy as np
from rouge_score import rouge_scorer
import re
import nltk
import spacy
from concurrent.futures import ProcessPoolExecutor


nlp = spacy.load('en_core_web_sm', disable=['ner'])

PARENTHESE_RE = re.compile(r'\(.*?\)')


def calculate_scores(prediction, target):
    prediction, _ = re.subn(r'\|\|\|\|', '|', prediction)
    prediction, _ = re.subn(r'{{', '{', prediction)
    prediction, _ = re.subn(r'}}}', '}', prediction)
    target, _ = re.subn(r'\|\|\|\|', '|', target)
    target, _ = re.subn(r'{{', '{', target)
    target, _ = re.subn(r'}}}', '}', target)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    smooth_method = nltk.translate.bleu_score.SmoothingFunction()

    target_depths = []
    target_sections = []
    target_relations = set()
    target_questions = []
    target_summaries = []
    target_first_summary = None
    current_depth = 1
    current_words = []
    current_question = []
    current_summary = []
    target_heads = {}
    stack = []
    now_question = False
    for word in target:
        if word == "{":
            now_question = True
            if target_first_summary is None:
                target_first_summary = "".join(current_summary).strip()
                current_summary = []
                current_question = []
                current_words = []
                stack.append(target_first_summary)
                target_heads[target_first_summary] = 'ROOT'
            elif "".join(current_question).strip():
                current_q = "".join(current_question).strip()
                current_s = "".join(current_summary).strip()
                target_questions.append(current_q)
                target_summaries.append(current_s)
                current_question = []
                current_summary = []
                if stack:
                    last_item = stack[-1]
                    if current_s not in target_heads:
                        target_heads[current_s] = last_item
                    target_relations.add((current_s, last_item))
                else:
                    if current_s not in target_heads:
                        target_heads[current_s] = 'ROOT'
                    target_relations.add((current_s, 'ROOT'))
                stack.append(current_s)
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                target_sections.append(current_qs)
                target_depths.append(current_depth if current_depth > 1 else 1)
                current_words = []

            current_depth += 1
        elif word == '|':
            now_question = True
            if target_first_summary is None:
                target_first_summary = "".join(current_summary).strip()
                current_summary = []
                current_question = []
                current_words = []
                target_heads[target_first_summary] = 'ROOT'
            elif "".join(current_question).strip():
                current_q = "".join(current_question).strip()
                current_s = "".join(current_summary).strip()
                target_questions.append(current_q)
                target_summaries.append(current_s)
                current_question = []
                current_summary = []
                if stack:
                    last_item = stack[-1]
                    if current_s not in target_heads:
                        target_heads[current_s] = last_item
                    target_relations.add((current_s, last_item))
                else:
                    if current_s not in target_heads:
                        target_heads[current_s] = 'ROOT'
                    target_relations.add((current_s, 'ROOT'))
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                target_sections.append(current_qs)
                target_depths.append(current_depth if current_depth > 1 else 1)
                current_words = []

        elif word == '}':
            now_question = True
            if target_first_summary is None:
                target_first_summary = "".join(current_summary).strip()
                current_summary = []
                current_question = []
                current_words = []
                target_heads[target_first_summary] = 'ROOT'
            elif "".join(current_question).strip():
                current_q = "".join(current_question).strip()
                current_s = "".join(current_summary).strip()
                target_questions.append(current_q)
                target_summaries.append(current_s)
                current_question = []
                current_summary = []
                if stack:
                    last_item = stack[-1]
                    if current_s not in target_heads:
                        target_heads[current_s] = last_item
                    target_relations.add((current_s, last_item))
                else:
                    if current_s not in target_heads:
                        target_heads[current_s] = 'ROOT'
                    target_relations.add((current_s, 'ROOT'))
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                target_sections.append(current_depth)
                target_depths.append(current_depth if current_depth > 1 else 1)
                current_words = []

            stack = stack[:-1]
            current_depth -= 1
        elif word == '?':
            current_question.append(word)
            now_question = False
        else:
            if now_question:
                current_question.append(word)
            else:
                current_summary.append(word)
            current_words.append(word)
    if "".join(current_words).strip():
        now_question = True
        if target_first_summary is None:
            target_first_summary = "".join(current_summary).strip()
            current_summary = []
            current_question = []
            current_words = []
            target_heads[target_first_summary] = 'ROOT'
        elif "".join(current_question).strip():
            current_q = "".join(current_question).strip()
            current_s = "".join(current_summary).strip()
            target_questions.append(current_q)
            target_summaries.append(current_s)
            current_question = []
            current_summary = []
            if stack:
                last_item = stack[-1]
                if current_s not in target_heads:
                    target_heads[current_s] = last_item
                target_relations.add((current_s, last_item))
            else:
                if current_s not in target_heads:
                    target_heads[current_s] = 'ROOT'
                target_relations.add((current_s, 'ROOT'))
        if "".join(current_words).strip():
            current_qs = "".join(current_words).strip()
            target_sections.append(current_depth)
            target_depths.append(current_depth if current_depth > 1 else 1)
            current_words = []

    prediction_depths = []
    prediction_sections = []
    prediction_relations = set()
    prediction_questions = []
    prediction_summaries = []
    prediction_first_summary = None
    current_depth = 1
    current_words = []
    current_question = []
    current_summary = []
    prediction_heads = {}
    stack = []
    for word in prediction:
        if word == "{":
            now_question = True
            if prediction_first_summary is None:
                prediction_first_summary = "".join(current_summary).strip()
                current_summary = []
                current_question = []
                current_words = []
                stack.append(prediction_first_summary)
                prediction_heads[prediction_first_summary] = 'ROOT'
            elif "".join(current_question).strip():
                current_q = "".join(current_question).strip()
                current_s = "".join(current_summary).strip()
                prediction_questions.append(current_q)
                prediction_summaries.append(current_s)
                current_question = []
                current_summary = []
                if stack:
                    last_item = stack[-1]
                    if current_s not in prediction_heads:
                        prediction_heads[current_s] = last_item
                    prediction_relations.add((current_s, last_item))
                else:
                    if current_s not in prediction_heads:
                        prediction_heads[current_s] = 'ROOT'
                    prediction_relations.add((current_s, 'ROOT'))
                stack.append(current_s)
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                prediction_sections.append(current_qs)
                prediction_depths.append(current_depth if current_depth > 1 else 1)
                current_words = []
            current_depth += 1
        elif word == '|':
            now_question = True
            if prediction_first_summary is None:
                prediction_first_summary = "".join(current_summary).strip()
                current_summary = []
                current_question = []
                current_words = []
                prediction_heads[prediction_first_summary] = 'ROOT'
            elif "".join(current_question).strip():
                current_q = "".join(current_question).strip()
                current_s = "".join(current_summary).strip()
                prediction_questions.append(current_q)
                prediction_summaries.append(current_s)
                current_question = []
                current_summary = []
                if stack:
                    last_item = stack[-1]
                    if current_s not in prediction_heads:
                        prediction_heads[current_s] = last_item
                    prediction_relations.add((current_s, last_item))
                else:
                    if current_s not in prediction_heads:
                        prediction_heads[current_s] = 'ROOT'
                    prediction_relations.add((current_s, 'ROOT'))
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                prediction_sections.append(current_qs)
                prediction_depths.append(current_depth if current_depth > 1 else 1)
                current_words = []

        elif word == '}':
            now_question = True
            if prediction_first_summary is None:
                prediction_first_summary = "".join(current_summary).strip()
                current_summary = []
                current_question = []
                current_words = []
                prediction_heads[prediction_first_summary] = 'ROOT'
            elif "".join(current_question).strip():
                current_q = "".join(current_question).strip()
                current_s = "".join(current_summary).strip()
                prediction_questions.append(current_q)
                prediction_summaries.append(current_s)
                current_question = []
                current_summary = []
                if stack:
                    last_item = stack[-1]
                    if current_s not in prediction_heads:
                        prediction_heads[current_s] = last_item
                    prediction_relations.add((current_s, last_item))
                else:
                    if current_s not in prediction_heads:
                        prediction_heads[current_s] = 'ROOT'
                    prediction_relations.add((current_s, 'ROOT'))
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                prediction_sections.append(current_depth)
                prediction_depths.append(current_depth if current_depth > 1 else 1)
                current_words = []
            stack = stack[:-1]
            current_depth -= 1
        elif word == '?':
            current_question.append(word)
            now_question = False
        else:
            if now_question:
                current_question.append(word)
            else:
                current_summary.append(word)
            current_words.append(word)
    if "".join(current_words).strip():
        now_question = True
        if prediction_first_summary is None:
            prediction_first_summary = "".join(current_summary).strip()
            current_summary = []
            current_question = []
            current_words = []
            prediction_heads[prediction_first_summary] = 'ROOT'
        elif "".join(current_question).strip():
            current_q = "".join(current_question).strip()
            current_s = "".join(current_summary).strip()
            prediction_questions.append(current_q)
            prediction_summaries.append(current_s)
            current_question = []
            current_summary = []
            if stack:
                last_item = stack[-1]
                if current_s not in prediction_heads:
                    prediction_heads[current_s] = last_item
                prediction_relations.add((current_s, last_item))
            else:
                if current_s not in prediction_heads:
                    prediction_heads[current_s] = 'ROOT'
                prediction_relations.add((current_s, 'ROOT'))
        if "".join(current_words).strip():
            current_qs = "".join(current_words).strip()
            prediction_sections.append(current_depth)
            prediction_depths.append(current_depth if current_depth > 1 else 1)
            current_words = []

    if target_heads:
        prediction2target = {}
        for prediction_title in prediction_heads.keys():
            title_scores = [(scorer.score(target_title, prediction_title), target_title) for target_title in
                            target_heads.keys()]
            title_scores = [(score['rouge1'].fmeasure + score['rouge2'].fmeasure, target_title) for score, target_title in
                            title_scores]
            aligned_title = sorted(title_scores, key=lambda x: x[0], reverse=True)[0]
            prediction2target[prediction_title] = aligned_title

        precision_nominator = 0
        for prediction_title, prediction_head in prediction_relations:
            if prediction_head != 'ROOT':
                target_title_score, mapped_target_title = prediction2target[prediction_title]
                target_head_score, mapped_target_head = prediction2target[prediction_head]

                while mapped_target_title != mapped_target_head and mapped_target_title != 'ROOT':
                    mapped_target_title = target_heads[mapped_target_title]

                precision_nominator += (target_title_score + target_head_score) / 4 * (
                            mapped_target_title == mapped_target_head)

        precision = precision_nominator / len(prediction_relations) if prediction_relations else 1
    else:
        precision = 0 if prediction_relations else 1

    if prediction_heads:
        target2prediction = {}
        for target_title in target_heads.keys():
            title_scores = [(scorer.score(prediction_title, target_title), prediction_title) for prediction_title in
                            prediction_heads.keys()]
            title_scores = [(score['rouge1'].fmeasure + score['rouge2'].fmeasure, prediction_title) for score, prediction_title in
                            title_scores]
            aligned_title = sorted(title_scores, key=lambda x: x[0], reverse=True)[0]
            target2prediction[target_title] = aligned_title

        recall_nominator = 0
        for target_title, target_head in target_relations:
            if target_head != 'ROOT':
                target_title_score, mapped_prediction_title = target2prediction[target_title]
                target_head_score, mapped_prediction_head = target2prediction[target_head]

                while mapped_prediction_title != mapped_prediction_head and mapped_prediction_title != 'ROOT':
                    mapped_prediction_title = prediction_heads[mapped_prediction_title]

                recall_nominator += (target_title_score + target_head_score) / 4 * (
                            mapped_prediction_title == mapped_prediction_head)

        recall = recall_nominator / len(target_relations) if target_relations else 1
    else:
        recall = 0 if target_relations else 1

    fscore = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0

    target_question = " ".join(target_questions)
    prediction_question = " ".join(prediction_questions)
    target_question_doc = nlp(target_question)
    prediction_question_doc = nlp(prediction_question)

    target_summary = target_first_summary + " " + " ".join(target_summaries)
    prediction_summary = prediction_first_summary + " " + " ".join(prediction_summaries)
    target_summary_doc = nlp(target_summary)
    prediction_summary_doc = nlp(prediction_summary)

    summary_score = scorer.score("\n".join([sent.text for sent in target_summary_doc.sents]), "\n".join([sent.text for sent in prediction_summary_doc.sents]))

    question_b4 = nltk.translate.bleu_score.sentence_bleu([[t.text.lower() for t in target_question_doc]],
                                                          [t.text.lower() for t in prediction_question_doc],
                                                          smoothing_function=smooth_method.method2,
                                                          weights=(0.25, 0.25, 0.25, 0.25))

    return summary_score, (precision, recall, fscore), question_b4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction')
    parser.add_argument('--target')
    args = parser.parse_args()

    with open(args.prediction) as f:
        predictions = [line.strip() for line in f]

    with open(args.target) as f:
        targets = [line.strip() for line in f]

    assert len(predictions) == len(targets)

    scores = []
    structure_scores = []
    b4s = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for prediction, target in zip(predictions, targets):
            futures.append(executor.submit(calculate_scores, prediction, target))
        for future in futures:
            score, structure_score, b4 = future.result()
            b4s.append(b4)
            scores.append(score)
            structure_scores.append(structure_score)

    print('Summary R-1', np.mean([score['rouge1'].fmeasure for score in scores]))
    print('Summary R-2', np.mean([score['rouge2'].fmeasure for score in scores]))
    print('Summary R-L', np.mean([score['rougeLsum'].fmeasure for score in scores]))
    print('Question B-4', np.mean(b4s))

    print('Hier Precision', np.mean([score[0] for score in structure_scores]))
    print('Hier Recall', np.mean([score[1] for score in structure_scores]))
    print('Hier F1', np.mean([score[2] for score in structure_scores]))


if __name__ == '__main__':
    main()