import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from treelstm import LayerwiseTreeLSTM
from treelstm_new import LayerwiseTreeLSTM_new

from span_rep import AvgSpanRepr, AttnSpanRepr

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

class SRLModel_attn(nn.Module):
    def __init__(self, encoder, num_layers, num_labels=1, just_last_layer=False,
                 use_proj=True, use_tag=False, tag_dim=64, proj_dim=256, pred_dim=256,
                 dropout=0.2, new_tree=True, comb_method='attn', use_overlap=False):
        super(SRLModel_attn, self).__init__()

        self.encoder = encoder
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.just_last_layer = just_last_layer
        self.use_proj = use_proj
        self.use_tag = use_tag
        self.tag_dim = tag_dim
        self.proj_dim = proj_dim
        self.pred_dim = pred_dim
        self.dropout = dropout
        self.comb_method = comb_method

        self.encoder_size = self.encoder.hidden_size

        self.span_net = AttnSpanRepr(self.encoder_size, use_proj=self.use_proj, proj_dim=self.proj_dim)

        if self.use_proj:
            self.pooled_dim = self.proj_dim
        else:
            self.pooled_dim = self.encoder_size
        if self.use_tag:
            self.pooled_dim += self.tag_dim
            self.tag_emb = nn.Embedding(num_embeddings=106, embedding_dim=self.tag_dim,
                                        padding_idx=105)
        if new_tree:
            self.Treenet = LayerwiseTreeLSTM_new(self.num_layers, self.pooled_dim, self.span_net, self.dropout, self.comb_method)
        else:
            self.Treenet = LayerwiseTreeLSTM(self.num_layers, self.pooled_dim, self.span_net, self.dropout, self.comb_method)

        if use_overlap:
            self.biaffine_net = Biaffine_overlap(self.num_labels, self.pooled_dim)
        else:
            self.biaffine_net = Biaffine(self.num_labels, self.pooled_dim)

        # self.training_criterion = nn.BCELoss()

    def forward(self, text, spans, child_rel, labels, tags, predicates, len_info):
        """
        :param text: [B, seq_len]
        :param spans: [B, 2 * num_spans]
        :param child_rel: [B, 3, num_rels]
        :param labels: [B, num_predicates, num_spans]
        :param tags: [B, num_spans]
        :param predicates: [B, num_predicates]
        :return:
        """
        if self.use_tag:
            tag_repr = self.tag_emb(tags)
        else:
            tag_repr = None
        encoded_input = self.encoder(text, just_last_layer=self.just_last_layer)

        span_repr = self.Treenet(spans, encoded_input, child_rel, tag_repr)

        # check = int((span_repr != span_repr).sum())
        # if (check > 0):
        #     logging.info("span_repr contains Nan")
        # else:
        #     logging.info("span_repr does not contain Nan, it might be other problem")

        pred_label, new_labels = self.biaffine_net(span_repr, predicates, labels, len_info)

        # label = torch.zeros_like(pred_label)
        # label.scatter_(1, new_labels.unsqueeze(dim=1), 1)
        # label = label.cuda().float()
        # loss = self.training_criterion(pred_label, label)
        # if self.training:
        #     return loss
        # else:
        #     return loss, pred_label, label
        return pred_label, new_labels


    def select_target_span(self, span_repr, span_index1, span_index2, labels, len_info):
        """
        span_repr: [B, num_spans, D]
        span_index: [B, num_instances]
        labels: [B, num_instances]
        len_info: [B]
        returns: s1 : [real_num_instances, D]
                s2: [real_num_instances, D]
                labels: [real_num_instances]
        """
        s1_repr, s2_repr, new_labels = [], [], []
        # s1, s1_l = span_index1
        # s2, s2_l = span_index2
        for i in range(len(len_info)):
            s1_repr.append(torch.index_select(span_repr[i], 0, span_index1[i][:len_info[i]]))
            s2_repr.append(torch.index_select(span_repr[i], 0, span_index2[i][:len_info[i]]))
            new_labels.append(labels[i][:len_info[i]])

        s1_repr = torch.cat(s1_repr, dim=0)
        s2_repr = torch.cat(s2_repr, dim=0)
        new_labels = torch.cat(new_labels, dim=0)

        return s1_repr, s2_repr, new_labels

    def get_other_params(self):
        core_encoder_param_names = set()
        for name, param in self.encoder.model.named_parameters():
            if param.requires_grad:
                core_encoder_param_names.add(name)

        other_params = []
        print("\nParams outside core transformer params:\n")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.size())
                other_params.append(param)
        print("\n")
        return other_params

    def get_core_params(self):
        return self.encoder.model.parameters()

class FFNN(nn.Module):
    def __init__(self, num_layers, feature_dim, dropout):
        super(FFNN, self).__init__()

        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.dropout = dropout

        self.linear_layers = nn.ModuleList([nn.Linear(self.feature_dim, self.feature_dim)
                                          for i in range(self.num_layers)])

    def forward(self, input):
        for i in range(self.num_layers):
            output = F.dropout(F.relu(self.linear_layers[i](input)), p=self.dropout)

        return output


class Biaffine(nn.Module):
    def __init__(self, num_labels, feature_dim, fflayers=2, dropout=0.5):
        super(Biaffine, self).__init__()

        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.fflayers = fflayers

        self.linear_p = FFNN(self.fflayers, self.feature_dim, self.dropout)
        self.linear_a = FFNN(self.fflayers, self.feature_dim, self.dropout)

        self.W_a = nn.Linear(self.feature_dim, 1, bias=False)
        self.W_p = nn.Linear(self.feature_dim, 1, bias=False)

        self.W1 = nn.Parameter(torch.FloatTensor(self.num_labels, self.feature_dim, self.feature_dim))
        self.W2 = nn.Parameter(torch.FloatTensor(2 * self.feature_dim, self.num_labels))
        self.b = nn.Parameter(torch.FloatTensor(1, self.num_labels))
        self.sigmoid = nn.Sigmoid()


    def forward(self, span_repr, predicates, labels, len_info):
        """
        :param span_repr: [B, num_spans, D]
        :param predicates: [B, num_predicates]
        :param labels: [B, num_predicates, num_spans]
        :param len_info: [B]
        :return:
        """
        stack_instance = []
        stack_label = []
        batch_size, num_predicates, num_spans = labels.size()
        for i in range(batch_size):
            mask = ~(predicates[i] == -1)
            label_mask = ~(labels[i] == -1)
            real_labels = torch.masked_select(labels[i], label_mask)     # [m * n]
            # null_labels = (real_labels == 66)   # [m * n]
            predicates_repr = torch.index_select(span_repr[i], 0 , predicates[i].masked_select(mask))       # [m, D]
            span_repr_trunc = span_repr[i][:len_info[i]]        # [n, D]
            n_predicates = predicates_repr.size(0)
            if n_predicates == 0:
                continue

            predicates_score = self.W_p(self.linear_p(predicates_repr))     # [m, 1]
            argument_score = self.W_a(self.linear_a(span_repr_trunc)).view(1, -1)       # [1, n]
            p_a_score = (predicates_score + argument_score).view(-1, 1)     # [m*n, 1]

            n_spans = span_repr_trunc.size(0)
            second = torch.cat((predicates_repr.unsqueeze(1).expand(-1, n_spans, -1),
                               span_repr_trunc.unsqueeze(0).expand(n_predicates, -1, -1)),
                               dim=-1)
            predicates_repr = predicates_repr.view(n_predicates, -1)
            span_repr_trunc = span_repr_trunc.view(n_spans, -1).transpose(0, 1).contiguous()
            first = torch.matmul(torch.matmul(predicates_repr, self.W1), span_repr_trunc)   # [C, m, n]
            first = first.view(-1, n_predicates * n_spans).transpose(0, 1).contiguous()      # [m * n, C]
            second = torch.matmul(second, self.W2).view(n_predicates * n_spans, -1)          # [m * n, C]
            rel_score = first + second + self.b             # [m * n, C]

            combine_score = rel_score + p_a_score           # [m * n, C]

            # combine_score = combine_score.masked_fill(null_labels.view(-1, 1), value=torch.tensor(0))

            combine_score[:, -1] = 0

            stack_instance.append(combine_score)
            stack_label.append(real_labels)

        stack_instance = torch.cat(stack_instance, dim=0)
        stack_label = torch.cat(stack_label)

        return stack_instance, stack_label

class Biaffine_overlap(nn.Module):
    def __init__(self, num_labels, feature_dim, fflayers=2, dropout=0.5):
        super(Biaffine_overlap, self).__init__()

        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.fflayers = fflayers

        self.linear_p = FFNN(self.fflayers, self.feature_dim, self.dropout)
        self.linear_a = FFNN(self.fflayers, self.feature_dim, self.dropout)

        self.W_a = nn.Linear(self.feature_dim, 1, bias=False)
        self.W_p = nn.Linear(self.feature_dim, 1, bias=False)

        self.W1 = nn.Parameter(torch.FloatTensor(self.num_labels, self.feature_dim, self.feature_dim))
        self.W2 = nn.Parameter(torch.FloatTensor(2 * self.feature_dim, self.num_labels))
        self.b = nn.Parameter(torch.FloatTensor(1, self.num_labels))
        self.sigmoid = nn.Sigmoid()


    def forward(self, span_repr, predicates, labels, long_mask):
        """
        :param span_repr: [B, num_spans, D]
        :param predicates: [B, num_predicates]
        :param labels: [B, num_predicates, num_spans]
        :param long_mask: [B, num_predicates, num_spans]
        :return:
        """
        stack_instance = []
        stack_label = []
        batch_size, num_predicates, num_spans = labels.size()
        for i in range(batch_size):
            mask = ~(predicates[i] == -1)
            # label_mask = ~(labels[i] == -1)
            # real_labels = torch.masked_select(labels[i], label_mask)     # [m * n]
            real_labels = torch.masked_select(labels[i], long_mask[i].bool())
            # null_labels = (real_labels == 66)   # [m * n]
            predicates_repr = torch.index_select(span_repr[i], 0 , predicates[i].masked_select(mask))       # [m, D]
            # span_repr_trunc = span_repr[i][:len_info[i]]        # [n, D]
            # span_repr_trunc = torch.masked_select(span_repr[i], long_mask[i])
            span_repr_trunc = span_repr[i]
            n_predicates = predicates_repr.size(0)
            if n_predicates == 0:
                continue

            predicates_score = self.W_p(self.linear_p(predicates_repr))     # [m, 1]
            argument_score = self.W_a(self.linear_a(span_repr_trunc)).view(1, -1)       # [1, n]
            p_a_score = (predicates_score + argument_score).view(-1, 1)     # [m*n, 1]

            n_spans = span_repr_trunc.size(0)
            second = torch.cat((predicates_repr.unsqueeze(1).expand(-1, n_spans, -1),
                               span_repr_trunc.unsqueeze(0).expand(n_predicates, -1, -1)),
                               dim=-1)
            predicates_repr = predicates_repr.view(n_predicates, -1)
            span_repr_trunc = span_repr_trunc.view(n_spans, -1).transpose(0, 1).contiguous()
            first = torch.matmul(torch.matmul(predicates_repr, self.W1), span_repr_trunc)   # [C, m, n]
            first = first.view(-1, n_predicates * n_spans).transpose(0, 1).contiguous()      # [m * n, C]
            second = torch.matmul(second, self.W2).view(n_predicates * n_spans, -1)          # [m * n, C]
            rel_score = first + second + self.b             # [m * n, C]

            combine_score = rel_score + p_a_score           # [m * n, C]

            # combine_score = combine_score.masked_fill(null_labels.view(-1, 1), value=torch.tensor(0))

            combine_score[:, -1] = 0
            score_mask = long_mask[i][:n_predicates].view(-1, 1).bool()
            combine_score = torch.masked_select(combine_score, score_mask)
            combine_score = combine_score.view(-1, self.num_labels)

            stack_instance.append(combine_score)
            stack_label.append(real_labels)

        stack_instance = torch.cat(stack_instance, dim=0)
        stack_label = torch.cat(stack_label)

        return stack_instance, stack_label
