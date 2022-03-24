import re
import argparse
import json


def _recursive_load_qs_section(section, current_id=0, parent_id=-1, depth=0):
    section_paragraphs = []
    child_id = current_id
    if section["section_title"]:
        section_paragraphs.append((" ".join(section["section_title"].split()),
                                   [" ".join(paragraph.split()) for paragraph in section["paragraphs"]], depth, current_id, parent_id))
        parent_id = current_id
        child_id += 1
    for subsection in section["subsections"]:
        child_paragraphs, ccid = _recursive_load_qs_section(subsection, child_id, parent_id, depth + 1)
        child_id = ccid
        section_paragraphs.extend(child_paragraphs)
    return section_paragraphs, child_id


def _load_qs(line):
    report = json.loads(line)
    document_paragraphs, _ = _recursive_load_qs_section(report["section"])
    first_question = report["questions"][0]["question"]
    return document_paragraphs, first_question


def pretty_print(line, first_question):
    line, _ = re.subn(r'\|\|\|\|', '|', line)
    line, _ = re.subn(r'{{', '{', line)
    line, _ = re.subn(r'}}}', '}', line)

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
    id_stack = []
    target_ids = []
    for word in line:
        if word == "{":
            now_question = True
            if target_first_summary is None:
                target_first_summary = "".join(current_summary).strip()
                id_stack = [1]
                current_summary = []
                current_question = []
                current_words = []
                stack.append(target_first_summary)
                target_heads[target_first_summary] = 'ROOT'
                id_stack.append(1)
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
                target_ids.append(".".join([str(x) for x in id_stack]))
                stack.append(current_s)
                id_stack.append(1)
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                target_sections.append(current_qs)
                target_depths.append(current_depth if current_depth > 1 else 1)
                # target_depths[" ".join(current_words)] = current_depth - 1 if current_depth - 1 > 1 else 1
                current_words = []

            current_depth += 1
        elif word == '|':
            now_question = True
            if target_first_summary is None:
                target_first_summary = "".join(current_summary).strip()
                id_stack = [1]
                current_summary = []
                current_question = []
                current_words = []
                stack.append(target_first_summary)
                target_heads[target_first_summary] = 'ROOT'
                id_stack[-1] = id_stack[-1] + 1
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
                target_ids.append(".".join([str(x) for x in id_stack]))
                id_stack[-1] = id_stack[-1] + 1
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                target_sections.append(current_qs)
                target_depths.append(current_depth if current_depth > 1 else 1)
                current_words = []

        elif word == '}':
            now_question = True
            if target_first_summary is None:
                target_first_summary = "".join(current_summary).strip()
                id_stack = [1]
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
            if "".join(current_words).strip():
                current_qs = "".join(current_words).strip()
                target_sections.append(current_depth)
                target_depths.append(current_depth if current_depth > 1 else 1)
                # target_depths[" ".join(current_words)] = current_depth - 1 if current_depth - 1 > 1 else 1
                current_words = []

            stack = stack[:-1]
            if id_stack:
                id_stack = id_stack[:-1]
            id_stack[-1] += 1
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
            id_stack = [1]
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
            target_ids.append(".".join([str(x) for x in id_stack]))
        if "".join(current_words).strip():
            current_qs = "".join(current_words).strip()
            target_sections.append(current_depth)
            target_depths.append(current_depth if current_depth > 1 else 1)
            # target_depths[" ".join(current_words)] = current_depth - 1 if current_depth - 1 > 1 else 1
            current_words = []

    assert len(target_summaries) == len(target_questions) == len(target_depths)

    output_text = ""
    output_text += f"|-- Q1: {first_question}\n    A1: {target_first_summary}\n"
    for target_question, target_summary, target_id in zip(target_questions, target_summaries, target_ids):
        depth = target_id.count(".")
        prefix_blank = "    " * depth
        output_text += f"{prefix_blank}|-- Q{target_id}: {target_question}\n{prefix_blank}    A{target_id}: {target_summary}\n"

    return output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_jsonl')
    parser.add_argument('--linearized_hierarchy')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    with open(args.source_jsonl, 'r') as f:
        source_jsonls = [line.strip() for line in f]

    with open(args.linearized_hierarchy, 'r') as f:
        linearized_hierarchies = [line.strip() for line in f]

    with open(args.output_file, 'w') as f:
        for i, (source_jsonl, linearized_hierarchy) in enumerate(zip(source_jsonls, linearized_hierarchies)):
            f.write(f"================   Sample {i: 6d}   =================\n")
            f.write(f"{pretty_print(linearized_hierarchy, _load_qs(source_jsonl)[1])}\n")


if __name__ == '__main__':
    main()
