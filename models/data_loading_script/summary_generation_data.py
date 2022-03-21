import datasets
import os
import json


def _get_question(current_question, depth=0):
    question_text = current_question["question"]
    summary = current_question["answer"]

    question_summary_pairs = [{'question': question_text, 'summary': summary, 'depth': depth}]

    for child_question in current_question["child_questions"]:
        question_summary_pairs.extend(_get_question(child_question, depth + 1))

    return question_summary_pairs


def _get_onehop_question(current_question, depth=0):
    question_text = current_question["question"]
    summary = current_question["answer"]

    question_summary_pairs = [{'question': question_text, 'summary': summary, 'depth': depth,
                               'child_pairs': [{'question': child_question['question'], 'summary': child_question['answer']} for child_question in current_question["child_questions"]]}]

    for child_question in current_question["child_questions"]:
        question_summary_pairs.extend(_get_onehop_question(child_question, depth + 1))

    return question_summary_pairs


def _recursive_load_qs_section(section, current_id=0, parent_id=-1, depth=0):
    section_paragraphs = []
    child_id = current_id
    if section["section_title"]:
        section_paragraphs.append((" ".join(section["section_title"].split()),
                                   [" ".join(paragraph.split()) for paragraph in section["paragraphs"]], depth, current_id, parent_id))
        parent_id = current_id
        child_id += 1
    for subsection in section["subsections"]:
        child_paragraphs, ccid = _recursive_load_section(subsection, child_id, parent_id, depth + 1)
        child_id = ccid
        section_paragraphs.extend(child_paragraphs)
    return section_paragraphs, child_id


def _load_qs(line):
    report = json.loads(line)
    document_paragraphs, _ = _recursive_load_qs_section(report["section"])
    question_summary_pairs = []
    for question in report["questions"]:
        question_summary_pairs.extend(_get_question(question))
    return document_paragraphs, question_summary_pairs, report["sample_id"]


def _load_onehop_qs(line):
    report = json.loads(line)
    document_paragraphs, _ = _recursive_load_qs_section(report["section"])
    question_summary_pairs = []
    for question in report["questions"]:
        question_summary_pairs.extend(_get_onehop_question(question))
    return document_paragraphs, question_summary_pairs, report["sample_id"]


def _recursive_load(section, keep_letter=False, current_id=0, parent_id=-1, depth=0):
    section_paragraphs = []
    child_id = current_id
    if section["section_title"] and (section["section_title"] != 'Letter' or (section["section_title"] == 'Letter' and keep_letter)):
        section_paragraphs.append((" ".join(section["section_title"].split()), [" ".join(paragraph.split()) for paragraph in section["paragraphs"]], depth, current_id, parent_id))
        child_id += 1
        for subsection in section["subsections"]:
            child_paragraphs, ccid = _recursive_load(subsection, keep_letter, child_id, current_id, depth + 1)
            child_id = ccid
            section_paragraphs.extend(child_paragraphs)
    else:
        for subsection in section["subsections"]:
            child_paragraphs, ccid = _recursive_load(subsection, keep_letter, child_id, parent_id, depth)
            child_id = ccid
            section_paragraphs.extend(child_paragraphs)

    return section_paragraphs, child_id


def _recursive_load_section(section, current_id=0, parent_id=-1, depth=0):
    section_paragraphs = []
    child_id = current_id
    section_paragraphs.append((" ".join(section["section_title"].split()), [" ".join(paragraph.split()) for paragraph in section["paragraphs"]], depth, current_id, parent_id))
    child_id += 1
    for subsection in section["subsections"]:
        child_paragraphs, ccid = _recursive_load_section(subsection, child_id, current_id, depth + 1)
        child_id = ccid
        section_paragraphs.extend(child_paragraphs)

    return section_paragraphs, child_id


def _load_gao_doc(filepath, no_rec=False):
    with open(filepath, encoding="utf-8") as f:
        report = json.load(f)
    document_paragraphs = []
    current_id = 0
    for section in report["report"]:
        paragraphs, pcid = _recursive_load(section, keep_letter=False, current_id=current_id)
        current_id = pcid
        document_paragraphs.extend(paragraphs)
    summary_paragraphs = []
    for section in report["highlight"]:
        if no_rec and section["section_title"] == "What GAO Recommends":
            continue
        summary_paragraphs.extend(section["paragraphs"])
    return document_paragraphs, summary_paragraphs


def _load_crs_doc(filepath):
    with open(filepath, encoding="utf-8") as f:
        report = json.load(f)
    document_paragraphs, _ = _recursive_load(report["reports"], keep_letter=True)
    summary_paragraphs = report["summary"]
    return document_paragraphs, summary_paragraphs


def _load_wiki(line):
    report = json.loads(line)
    document_paragraphs, _ = _recursive_load(report["report"], keep_letter=True)
    summary_paragraphs = report["summary"]
    return document_paragraphs, summary_paragraphs, report['id']


class FullSummaryGenerationConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FullSummaryGenerationDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        FullSummaryGenerationConfig(
            name="gov_report",
            version=VERSION,
            description="gov_report"
        ),
        FullSummaryGenerationConfig(
            name="qs_hierarchy_fq",
            version=VERSION,
            description="qs_hierarchy_fq"
        ),
        FullSummaryGenerationConfig(
            name="qs_hierarchy_qg",
            version=VERSION,
            description="qs_hierarchy_qg"
        ),
        FullSummaryGenerationConfig(
            name="wiki_bio_sum",
            version=VERSION,
            description="wiki_bio_sum"
        ),
    ]

    def _info(self):
        if self.config.name == 'gov_report':
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_paragraphs": [datasets.Value("string")],
                    "section_paragraph_ends": [datasets.Value("int32")],
                    "section_depths": [datasets.Value("int32")],
                    "section_parent_ids": [datasets.Value("int32")],
                    "section_titles": [datasets.Value("string")],
                    "summary": datasets.Value("string")
                }
            )
        elif self.config.name.startswith('wiki_bio_sum'):
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_paragraphs": [datasets.Value("string")],
                    "section_paragraph_ends": [datasets.Value("int32")],
                    "section_depths": [datasets.Value("int32")],
                    "section_parent_ids": [datasets.Value("int32")],
                    "section_titles": [datasets.Value("string")],
                    "summary": datasets.Value("string")
                }
            )
        elif self.config.name == 'qs_hierarchy_fq':
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_paragraphs": [datasets.Value("string")],
                    "section_paragraph_ends": [datasets.Value("int32")],
                    "section_depths": [datasets.Value("int32")],
                    "section_parent_ids": [datasets.Value("int32")],
                    "section_titles": [datasets.Value("string")],
                    "first_question": datasets.Value("string"),
                    "first_summary": datasets.Value("string"),
                    "summary_paragraphs": [datasets.Value("string")],
                    "summary_depths": [datasets.Value("int32")]
                }
            )
        elif self.config.name == 'qs_hierarchy_qg':
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_paragraphs": [datasets.Value("string")],
                    "section_paragraph_ends": [datasets.Value("int32")],
                    "section_depths": [datasets.Value("int32")],
                    "section_parent_ids": [datasets.Value("int32")],
                    "section_titles": [datasets.Value("string")],
                    "first_question": datasets.Value("string"),
                    "first_summary": datasets.Value("string"),
                    "summary_paragraphs": [datasets.Value("string")],
                }
            )
        else:
            raise ValueError

        return datasets.DatasetInfo(
            description="summary dataset",
            features=features,
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split_name": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split_name": "valid"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split_name": "test"
                }
            )
        ]

    def _generate_examples(self, split_name):
        if self.config.name == 'wiki_bio_sum':
            with open(os.path.join(self.config.data_dir, 'wikibiosum', f'{split_name}.jsonl')) as f:
                for line in f:
                    document_sections, summary_paragraphs, wiki_id = _load_wiki(line)
                    summary = " ".join(summary_paragraphs)

                    _id = wiki_id

                    section_paragraph_ends = []
                    section_depths = []
                    document_paragraphs = []
                    current_end = 0
                    section_titles = []
                    parent_ids = []
                    for section_title, document_section, section_depth, current_id, parent_id in document_sections:
                        paragraphs = document_section
                        section_titles.append(section_title)

                        current_end += len(paragraphs)

                        section_paragraph_ends.append(current_end)
                        section_depths.append(section_depth)
                        document_paragraphs.extend(paragraphs)
                        parent_ids.append(parent_id)

                    if document_paragraphs:
                        yield _id, {
                            "id": _id,
                            "document_paragraphs": document_paragraphs,
                            "section_paragraph_ends": section_paragraph_ends,
                            "section_depths": section_depths,
                            "section_parent_ids": parent_ids,
                            "section_titles": section_titles,
                            "summary": summary
                        }
        elif self.config.name == 'qs_hierarchy_fq':
            with open(os.path.join(self.config.data_dir, 'gov-report-qs', f'{split_name}.jsonl')) as f:
                for line_i, line in enumerate(f):
                    document_sections, question_summary_pairs, sample_id = _load_qs(line)
                    first_question = question_summary_pairs[0]["question"]
                    first_summary = question_summary_pairs[0]["summary"]
                    summary_paragraphs = [question_summary_pair["question"] + " " + question_summary_pair["summary"] for
                                          question_summary_pair in question_summary_pairs[1:]]
                    summary_depths = [question_summary_pair["depth"] for question_summary_pair in
                                      question_summary_pairs[1:]]

                    _id = f'{sample_id}_{line_i}'

                    section_paragraph_ends = []
                    section_depths = []
                    document_paragraphs = []
                    current_end = 0
                    section_titles = []
                    parent_ids = []
                    for section_title, document_section, section_depth, current_id, parent_id in document_sections:
                        paragraphs = document_section
                        section_titles.append(section_title)

                        current_end += len(paragraphs)

                        section_paragraph_ends.append(current_end)
                        section_depths.append(section_depth)
                        document_paragraphs.extend(paragraphs)
                        parent_ids.append(parent_id)

                    if document_paragraphs:
                        yield _id, {
                            "id": _id,
                            "document_paragraphs": document_paragraphs,
                            "section_paragraph_ends": section_paragraph_ends,
                            "section_depths": section_depths,
                            "section_parent_ids": parent_ids,
                            "section_titles": section_titles,
                            "first_question": first_question,
                            "first_summary": first_summary,
                            "summary_paragraphs": summary_paragraphs,
                            "summary_depths": summary_depths
                        }
        elif self.config.name == 'qs_hierarchy_qg':
            with open(os.path.join(self.config.data_dir, 'gov-report-qs', f'{split_name}.jsonl')) as f:
                for line_i, line in enumerate(f):
                    document_sections, question_summary_pairs, sample_id = _load_onehop_qs(line)

                    _id = f'{sample_id}_{line_i}'

                    section_paragraph_ends = []
                    section_depths = []
                    document_paragraphs = []
                    current_end = 0
                    section_titles = []
                    parent_ids = []
                    for section_title, document_section, section_depth, current_id, parent_id in document_sections:
                        paragraphs = document_section
                        section_titles.append(section_title)

                        current_end += len(paragraphs)

                        section_paragraph_ends.append(current_end)
                        section_depths.append(section_depth)
                        document_paragraphs.extend(paragraphs)
                        parent_ids.append(parent_id)

                    if document_paragraphs:
                        for qi, question_summary_pair in enumerate(question_summary_pairs):
                            if question_summary_pair["question"] and question_summary_pair["summary"] and question_summary_pair["child_pairs"]:
                                summary_paragraphs = [
                                    pair["question"] for
                                    pair in question_summary_pair["child_pairs"]]
                                yield f'{_id}_{qi}', {
                                    "id": f'{_id}_{qi}',
                                    "document_paragraphs": document_paragraphs,
                                    "section_paragraph_ends": section_paragraph_ends,
                                    "section_depths": section_depths,
                                    "section_parent_ids": parent_ids,
                                    "section_titles": section_titles,
                                    "first_question": question_summary_pair["question"],
                                    "first_summary": question_summary_pair["summary"],
                                    "summary_paragraphs": summary_paragraphs,
                                }
        elif self.config.name == 'gov_report':
            gao_split_file = os.path.join(self.config.data_dir, "gov-report", "split_ids", f'gao_{split_name}.ids')
            crs_split_file = os.path.join(self.config.data_dir, "gov-report", "split_ids", f'crs_{split_name}.ids')
            document_dir = os.path.join(self.config.data_dir, "gov-report")

            with open(gao_split_file) as f:
                gao_split_ids = [line.strip() for line in f]

            with open(crs_split_file) as f:
                crs_split_ids = [line.strip() for line in f]

            for gao_split_id in gao_split_ids:
                document_sections, summary_paragraphs = _load_gao_doc(
                    os.path.join(document_dir, 'gao', f'{gao_split_id}.json'), no_rec=False)

                summary = " ".join(summary_paragraphs)

                _id = 'GAO_' + gao_split_id

                section_paragraph_ends = []
                section_depths = []
                document_paragraphs = []
                current_end = 0
                section_titles = []
                parent_ids = []
                for section_title, document_section, section_depth, current_id, parent_id in document_sections:
                    paragraphs = document_section
                    section_titles.append(section_title)

                    current_end += len(paragraphs)

                    section_paragraph_ends.append(current_end)
                    section_depths.append(section_depth)
                    document_paragraphs.extend(paragraphs)
                    parent_ids.append(parent_id)

                if document_paragraphs:
                    yield _id, {
                        "id": _id,
                        "document_paragraphs": document_paragraphs,
                        "section_paragraph_ends": section_paragraph_ends,
                        "section_depths": section_depths,
                        "section_parent_ids": parent_ids,
                        "section_titles": section_titles,
                        "summary": summary
                    }

            for crs_split_id in crs_split_ids:
                document_sections, summary_paragraphs = _load_crs_doc(
                    os.path.join(document_dir, 'crs', f'{crs_split_id}.json'))
                summary = " ".join(summary_paragraphs)

                _id = 'CRS_' + crs_split_id

                section_paragraph_ends = []
                section_depths = []
                document_paragraphs = []
                current_end = 0
                section_titles = []
                parent_ids = []
                for section_title, document_section, section_depth, current_id, parent_id in document_sections:
                    paragraphs = document_section
                    section_titles.append(section_title)

                    current_end += len(paragraphs)

                    section_paragraph_ends.append(current_end)
                    section_depths.append(section_depth)
                    document_paragraphs.extend(paragraphs)
                    parent_ids.append(parent_id)

                if document_paragraphs:
                    yield _id, {
                        "id": _id,
                        "document_paragraphs": document_paragraphs,
                        "section_paragraph_ends": section_paragraph_ends,
                        "section_depths": section_depths,
                        "section_parent_ids": parent_ids,
                        "section_titles": section_titles,
                        "summary": summary
                    }
