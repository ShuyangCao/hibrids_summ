import argparse
from rouge_score import rouge_scorer
import spacy
import numpy as np
import nltk
from concurrent.futures import ProcessPoolExecutor


nlp = spacy.load('en_core_web_sm', disable=['ner'])

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


def eval_one(prediction, target):
    prediction_qs = prediction.split("||||")
    prediction_q = " ".join(prediction_qs)

    target_qs = target.split("||||")
    target_q = " ".join(target_qs)

    pred_q_doc = nlp(prediction_q)
    target_q_doc = nlp(target_q)

    score = scorer.score("\n".join([sent.text for sent in target_q_doc.sents]), "\n".join([sent.text for sent in pred_q_doc.sents]))

    smooth_method = nltk.translate.bleu_score.SmoothingFunction()
    question_b4 = nltk.translate.bleu_score.sentence_bleu([[t.text.lower() for t in target_q_doc]],
                                                          [t.text.lower() for t in pred_q_doc],
                                                          smoothing_function=smooth_method.method2,
                                                          weights=(0.25, 0.25, 0.25, 0.25))

    return (score['rouge1'].fmeasure, score['rouge2'].fmeasure, score['rougeLsum'].fmeasure), question_b4


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
    print(len(predictions), len(targets))

    scores = []
    bleu4_scores = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for prediction, target in zip(predictions, targets):
            futures.append(executor.submit(eval_one, prediction, target))
        for future in futures:
            rs, b4 = future.result()

            scores.append(rs)
            bleu4_scores.append(b4)

    print('R1', np.mean([score[0] for score in scores]))
    print('R2', np.mean([score[1] for score in scores]))
    print('RL', np.mean([score[2] for score in scores]))
    print('B4', np.mean(bleu4_scores))


if __name__ == '__main__':
    main()