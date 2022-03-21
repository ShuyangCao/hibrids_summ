import argparse
from rouge_score import rouge_scorer
import spacy
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor


nlp = spacy.load('en_core_web_sm', disable=['ner'])

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def eval_one(prediction, target):
    pred_doc = nlp(prediction)
    target_doc = nlp(target)

    sent_prediction = "\n".join([sent.text for sent in pred_doc.sents])
    sent_target = "\n".join([sent.text for sent in target_doc.sents])

    score = scorer.score(sent_target, sent_prediction)
    return score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeLsum'].fmeasure


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
    with ProcessPoolExecutor() as executor:
        futures = []
        for prediction, target in zip(predictions, targets):
            futures.append(executor.submit(eval_one, prediction, target))
        for future in futures:
            score = future.result()
            scores.append(score)

    print('R1', np.mean([score[0] for score in scores]))
    print('R2', np.mean([score[1] for score in scores]))
    print('RL', np.mean([score[2] for score in scores]))


if __name__ == '__main__':
    main()