import time
import json

import torch
import torch.utils.data

def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
#             new_item = {}
            item = json.loads(line)
            data.append(item)
    return data


class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_path, encoder):
        self.example_list = readJson(data_path)
        self.size = len(self.example_list)
        self.encoder = encoder

    def __getitem__(self, index):
        item = self.example_list[index]
        src_text = item['words']
        text, subword_to_word_idx = self.encoder.tokenize(src_text, get_subword_indices=True)

        new_spans = []
        span_list = []
        for span in item['spans']:
            sta, end = span
            new_span = [int(sta), int(end)]
            span_list.append(new_span)
            span_index = self.get_tokenized_span_indices(
                subword_to_word_idx, new_span)
            sta, end = span_index
            new_spans.append(int(sta))
            new_spans.append(int(end))

        tags = []
        for tag in item['span_label']:
            tags.append(int(tag))

        child_rel = []
        for p, children in item['child_rel'].items():
            child_rel.append(int(p))
            lc, rc = children
            child_rel.append(int(lc))
            child_rel.append(int(rc))

        predicate_index = []
        labels = []
        for target in item['targets']:
            if target["span1"] not in predicate_index:
                label = [66] * len(tags)
                predicate_index.append(target["span1"])
                labels.append(label)

        for target in item['targets']:
            labels[predicate_index.index(target['span1'])][target['span2']] = target['label']


        return text, new_spans, child_rel, tags, predicate_index, labels

    def __len__(self):
        return self.size

    @staticmethod
    def get_tokenized_span_indices(subword_to_word_idx, orig_span_indices):
        orig_start_idx, orig_end_idx = orig_span_indices
        start_idx = subword_to_word_idx.index(orig_start_idx)
        # Search for the index of the last subword
        end_idx = len(subword_to_word_idx) - 1 - subword_to_word_idx[::-1].index(orig_end_idx - 1)
        return [start_idx, end_idx]


def new_collate_fn(samples):
    text, spans, child_rel, tags, predicate_index, labels = map(list, zip(*samples))
    batch_size = len(text)
    text_len = [len(s) for s in text]
    max_seq_len = max(text_len)
    max_num_spans = max([len(s) for s in spans])
    max_num_tags = max([len(s) for s in tags])
    assert max_num_spans == max_num_tags * 2
    max_num_rels = max([len(s) for s in child_rel])
    len_info = [len(s) for s in tags]
    max_predicates = max([len(s) for s in predicate_index])
    # max_instances = max(len_info)

    new_text = []
    new_spans = []
    new_rels = []
    new_tags = []
    new_predicate_index = []
    new_labels = []
    for i in range(batch_size):
        # if len(predicate_index[i]) == 0:
        #     continue
        label_for_one_instance = []
        new_text.append(text[i] + [0] * (max_seq_len - len(text[i])))
        new_spans.append(spans[i] + [-1] * (max_num_spans - len(spans[i])))
        new_rels.append(child_rel[i] + [-1] * (max_num_rels - len(child_rel[i])))
        new_tags.append(tags[i] + [105] * (max_num_tags - len(tags[i])))
        new_predicate_index.append(predicate_index[i] + [-1] * (max_predicates - len(predicate_index[i])))
        # new_span2_index.append(span2_index[i] + [-1] * (max_instances - len_info[i]))
        for j in range(len(labels[i])):
            label_for_one_instance.append(labels[i][j] + [-1] * (max_num_tags - len(labels[i][j])))
        for m in range(max_predicates - len(labels[i])):
            label_for_one_instance.append([-1] * max_num_tags)
        new_labels.append(label_for_one_instance)

    new_text = torch.LongTensor(new_text)
    text_len = torch.LongTensor(text_len)
    new_spans = torch.LongTensor(new_spans)
    new_rels = torch.LongTensor(new_rels)
    new_tags = torch.LongTensor(new_tags)
    new_predicate_index = torch.LongTensor(new_predicate_index)
    new_labels = torch.LongTensor(new_labels)
    len_info = torch.LongTensor(len_info)

    return (new_text, text_len), new_spans, new_rels, new_tags, new_predicate_index, new_labels, len_info