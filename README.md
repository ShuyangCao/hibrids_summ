# HIBRIDS

Code for ACL 2022 paper "HIBRIDS: Attention with Hierarchical Biases for Structure-aware Long Document Summarization".

-------

## Data

- [GovReport-QS and GovReport](https://gov-report-data.github.io/)
- [WikiBioSum](https://drive.google.com/drive/folders/1vjT_eTWjRmZmlayP-9zCzDEH-4qbmCJ4?usp=sharing)

-------

## Model

### Requirements

```shell
# Suggested: create a virtual environment
conda create -n hibrids python=3.8
conda activate hibrids

# Fairseq (commit f34abc)
# Newer versions might also work
mkdir lib
cd lib
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout f34abc
pip install -e .
cd ../..

# Other requirements
# Install after Fairseq because fairseq overrides torch version
pip install -r requirements.txt

# Fairseq C extensions
cd lib/fairseq
python setup.py build_ext --inplace
cd ../..

# Apex for fp16 training
cd lib
git clone --recurse-submodules https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
cd ../..
```

### Download Checkpoints & Set Environment Variables

For trained checkpoints, you can download from [here](https://drive.google.com/drive/folders/1vjT_eTWjRmZmlayP-9zCzDEH-4qbmCJ4?usp=sharing).

Please organize data and trained models as follows:

```
EXPDIR
|- data
    |- gov-report-qs
    |- wikibiosum
    |- gov-report
|- trained_models
    |- qs_hierarchy_fq
        |- <model_name>
            |- checkpoint_best.pt
    |- qs_hierarchy_qg
    |- wiki_bio_sum
    |- gov_report
```

Set environment variables.

```shell
export EXPDIR=<path_to_experiment_directory>
export CODEDIR=<path_to_this_repo>
```

### Decode with Trained Models

By default, the decoded outputs are saved in `$EXPDIR/decode_outputs/<dataset>/<model>/generated_predictions.txt`.
You can change it by setting the `--output_dir` argument.

#### Hierarchical Bias on Encoder

```shell
cd models/hierarchical_bias

# QS Hierarchy given the First Question
python decode_qs_hierarchy_fq.py --model_dir $EXPDIR/trained_models/qs_hierarchy_fq/hierarchical_bias

# Child Question Generation
python decode_qs_hierarchy_qg.py --model_dir $EXPDIR/trained_models/qs_hierarchy_qg/hierarchical_bias

# WikiBioSum
python decode_wiki.py --model_dir $EXPDIR/trained_models/wiki_bio_sum/hierarchical_bias

# Gov Report
python decode_gov_report.py --model_dir $EXPDIR/trained_models/gov_report/hierarchical_bias
```

#### Hierarchical Bias on Decoder

```shell
cd models/hierarchical_bias_decoder

# QS Hierarchy given the First Question
python decode_qs_hierarchy_fq.py --model_dir $EXPDIR/trained_models/qs_hierarchy_fq/hierarchical_bias_decoder

# Child Question Generation
python decode_qs_hierarchy_qg.py --model_dir $EXPDIR/trained_models/qs_hierarchy_qg/hierarchical_bias_decoder

# WikiBioSum
python decode_wiki.py --model_dir $EXPDIR/trained_models/wiki_bio_sum/hierarchical_bias_decoder

# Gov Report
python decode_gov_report.py --model_dir $EXPDIR/trained_models/gov_report/hierarchical_bias_decoder
```

### Train Models

#### Hierarchical Bias on Encoder

```shell
cd models/configs/hierarchical_bias

# QS Hierarchy given the First Question
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name qs_hierarchy_fq.yaml

# Child Question Generation
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name qs_hierarchy_qg.yaml

# WikiBioSum
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name wiki_bio_sum.yaml

# Gov Report
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name gov_report.yaml
```

#### Hierarchical Bias on Decoder

```shell
cd models/configs/hierarchical_bias_decoder

# QS Hierarchy given the First Question
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name qs_hierarchy_fq.yaml

# Child Question Generation
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name qs_hierarchy_qg.yaml

# WikiBioSum
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name wiki_bio_sum.yaml

# Gov Report
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-hydra-train --config-dir . --config-name gov_report.yaml
```

### Evaluation

```shell
cd evaluation

# QS Hierarchy given the First Question
python eval_qs_hierarchy_fq.py \
  --prediction $EXPDIR/decode_outputs/qs_hierarchy_fq/hierarchical_bias/generated_predictions.txt \
  --target $EXPDIR/decode_outputs/qs_hierarchy_fq/hierarchical_bias/targets.txt
 
# Child Question Generation
python eval_qg.py \
  --prediction $EXPDIR/decode_outputs/qs_hierarchy_qg/hierarchical_bias/generated_predictions.txt \
  --target $EXPDIR/decode_outputs/qs_hierarchy_qg/hierarchical_bias/targets.txt

# WikiBioSum
python eval_summary.py \
  --prediction $EXPDIR/decode_outputs/wiki_bio_sum/hierarchical_bias/generated_predictions.txt \
  --target $EXPDIR/decode_outputs/wiki_bio_sum/hierarchical_bias/targets.txt

# Gov Report
python eval_summary.py \
  --prediction $EXPDIR/decode_outputs/gov_report/hierarchical_bias/generated_predictions.txt \
  --target $EXPDIR/decode_outputs/gov_report/hierarchical_bias/targets.txt
```
