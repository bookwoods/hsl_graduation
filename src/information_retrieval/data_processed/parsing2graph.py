import os
import dgl
import torch
from transformers import AutoTokenizer
from ltp import LTP
from src.config import hparams

class P2G_single:
    def __init__(self, dataset, pretrained_model, parsing_model, max_len):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.ltp = LTP(parsing_model)
        self.max_len = max_len
        self.sents1, self.sents2, self.labels = self.len_limit()
        self.encoded1, self.graph1, self.g1_num_nodes, self.encoded2, self.graph2, self.g2_num_nodes = self.read_data()

    def len_limit(self):
        sentence1 = self.dataset["sentence1"]
        sentence2 = self.dataset["sentence2"]
        labels = self.dataset["score"]
        sents1 = []
        sents2 = []
        new_labels = []
        for sent1, sent2, label in zip(sentence1, sentence2, labels):
            tokens1 = self.tokenizer.tokenize(sent1)
            tokens2 = self.tokenizer.tokenize(sent2)
            if len(tokens1) < 127 and len(tokens2) < 127:
                sents1.append(sent1)
                sents2.append(sent2)
                new_labels.append(label)
        return sents1, sents2, new_labels



    def read_data(self):
        graph1 = []
        graph2 = []
        g1_num_nodes = []
        g2_num_nodes = []
        for sent1, sent2 in zip(self.sents1, self.sents2):
            g1, nodes_num1 = self.build_graph(sent1)
            graph1.append(g1)
            g1_num_nodes.append(nodes_num1)
            g2, nodes_num2 = self.build_graph(sent2)
            graph2.append(g2)
            g2_num_nodes.append(nodes_num2)
        encoded1 = self.tokenizer(self.sents1, return_tensors='pt', padding="max_length", truncation=True,
                                  max_length=self.max_len)
        encoded2 = self.tokenizer(self.sents2, return_tensors='pt', padding="max_length", truncation=True,
                                  max_length=self.max_len)
        return encoded1, graph1, g1_num_nodes, encoded2, graph2, g2_num_nodes


    def build_graph(self, sentence):
        # 字粒度分词
        word_tokens = self.tokenizer.tokenize(sentence)
        # BERT分词时可能会在中英文词中添加#进行截断
        for i, token in enumerate(word_tokens):
            if "#" in token:
                word_tokens[i] = token.replace("#", "")
        sent = "".join(word_tokens)
        result = self.ltp.pipeline(sent, tasks=["cws", "dep", "sdp"])
        # 词粒度分词
        words_tokens = result.cws
        dep = result.dep  # 依存树
        sdp = result.sdp  # 语义依存树

        rels, all_labels = self.parsing(dep)
        g_data = {}
        g_data[('token', 'dep', 'token')] = (rels['dep']['head'], rels['dep']['tail'])
        g_data[('token', '-dep', 'token')] = (rels['-dep']['head'], rels['-dep']['tail'])
        g_data[('token', 'loop', 'token')] = (rels['loop']['head'], rels['loop']['tail'])
        g = dgl.heterograph(g_data)
        nodes_num = g.num_nodes()
        # 分词掩码映射
        mask = self.token2mask(word_tokens, words_tokens)
        # 节点的特征掩码
        g.ndata['mask'] = mask

        return g, nodes_num


    def parsing(self, dep_parsing):
        nodes = dep_parsing['head']
        labels = dep_parsing['label']
        all_labels = list(set(labels))
        all_labels.remove('HED')
        rels = {}
        rels['dep'] = {'head': [], 'tail': []}
        rels['-dep'] = {'head': [], 'tail': []}
        rels['loop'] = {'head': [], 'tail': []}
        for idx, head in enumerate(nodes):
            if labels[idx] != 'HED':
                rels['dep']['head'].append(idx)
                rels['dep']['tail'].append(head-1)
                rels['-dep']['head'].append(head-1)
                rels['-dep']['tail'].append(idx)
            rels['loop']['head'].append(idx)
            rels['loop']['tail'].append(idx)
        return rels, all_labels

    def token2mask(self, tokens1, tokens2):
        mask = torch.zeros((len(tokens2), self.max_len))
        end_index = 0
        for idx, word2 in enumerate(tokens2):
            for i, word1 in enumerate(tokens1):
                if word2.startswith(word1) and i >= end_index:
                    start_index = i  # 添加起始索引
                if word2.endswith(word1) and i >= end_index:
                    end_index = i  # 添加结束索引
                    break
            # 分词时首位添加[CLS]，索引需要+1
            start_index = start_index + 1
            end_index = end_index + 1
            # 判断起始位与终止位是否相同
            if start_index == end_index:
                current_index = [start_index]
            else:
                current_index = list(range(start_index, end_index + 1))
            mask[idx][current_index] = 1
        return mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids1 = self.encoded1["input_ids"][idx]
        attention_mask1 = self.encoded1["attention_mask"][idx]
        graph1 = self.graph1[idx]
        g1_num_nodes = torch.tensor(self.g1_num_nodes[idx])
        input_ids2 = self.encoded2["input_ids"][idx]
        attention_mask2 = self.encoded2["attention_mask"][idx]
        graph2 = self.graph2[idx]
        g2_num_nodes = torch.tensor(self.g2_num_nodes[idx])
        label = torch.tensor(self.labels[idx])
        return input_ids1, attention_mask1, graph1, g1_num_nodes, input_ids2, attention_mask2, graph2, g2_num_nodes, label


class P2G_double:
    def __init__(self, dataset, pretrained_model, parsing_model, max_len):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.ltp = LTP(parsing_model)
        self.max_len = max_len
        self.encoded, self.graph, self.g_num_nodes, self.new_labels = self.main()

    def main(self):
        sentence1 = self.dataset["sentence1"]
        sentence2 = self.dataset["sentence2"]
        labels = self.dataset["score"]
        encoded = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": []
        }
        graph = []
        new_labels = []
        g_num_nodes = []
        for sent1, sent2, label in zip(sentence1, sentence2, labels):
            sent1_tokens1 = self.tokenizer.tokenize(sent1)
            sent2_tokens1 = self.tokenizer.tokenize(sent2)
            seq_tokens = ['[CLS]'] + sent1_tokens1 + ['[SEP]'] + sent2_tokens1 + ['[SEP]']
            # 合并后的seq_tokens不得大于max_len
            if len(seq_tokens) > self.max_len:
                continue
            g, nodes_num = self.build_graph(sent1_tokens1, sent2_tokens1)
            if g is not None:
                graph.append(g)
                g_num_nodes.append(nodes_num)
                new_labels.append(label)
            else:
                continue
            token_type_ids = [0] * (len(sent1_tokens1) + 2) + [1] * (len(sent2_tokens1) + 1)
            input_ids = self.tokenizer.convert_tokens_to_ids(seq_tokens)
            # 根据max_len与seq的长度产生填充序列
            padding = [0] * (self.max_len - len(seq_tokens))
            input_ids = input_ids + padding
            attention_mask = [1] * len(seq_tokens) + padding
            token_type_ids = token_type_ids + padding
            encoded["input_ids"].append(input_ids)
            encoded["token_type_ids"].append(token_type_ids)
            encoded["attention_mask"].append(attention_mask)
        return encoded, graph, g_num_nodes, new_labels

    def build_graph(self, sent1_tokens1, sent2_tokens1):
        # 分词器对于较长的词会插入#截断处理
        for i, token in enumerate(sent1_tokens1):
            if "#" in token:
                sent1_tokens1[i] = token.replace("#", "")
        new_sent1 = "".join(sent1_tokens1)
        for i, token in enumerate(sent2_tokens1):
            if "#" in token:
                sent2_tokens1[i] = token.replace("#", "")
        new_sent2 = "".join(sent2_tokens1)

        result1 = self.ltp.pipeline(new_sent1, tasks=["cws", "dep", "sdp", "sdpg"])
        result2 = self.ltp.pipeline(new_sent2, tasks=["cws", "dep", "sdp", "sdpg"])
        sent1_tokens2 = result1.cws
        dep1 = result1.dep
        sent2_tokens2 = result2.cws
        dep2 = result2.dep
        # 分词掩码映射
        mask = self.token2mask(sent1_tokens1, sent1_tokens2, sent2_tokens1, sent2_tokens2)
        # 构图
        g1 = self.parsing(dep1)
        g2 = self.parsing(dep2)
        graph = dgl.batch([g1, g2])  # 子图合并
        nodes_num = graph.num_nodes()
        # 节点的特征掩码
        graph.ndata['mask'] = mask
        return graph, nodes_num



    def parsing(self, dep_parsing):
        nodes = dep_parsing['head']
        labels = dep_parsing['label']
        all_labels = list(set(labels))
        all_labels.remove('HED')
        rels = {}
        rels['dep'] = {'head': [], 'tail': []}
        rels['-dep'] = {'head': [], 'tail': []}
        rels['loop'] = {'head': [], 'tail': []}
        for idx, head in enumerate(nodes):
            if labels[idx] != 'HED':
                rels['dep']['head'].append(idx)
                rels['dep']['tail'].append(head-1)
                rels['-dep']['head'].append(head-1)
                rels['-dep']['tail'].append(idx)
            rels['loop']['head'].append(idx)
            rels['loop']['tail'].append(idx)
        g_data = {}
        g_data[('token', 'dep', 'token')] = (rels['dep']['head'], rels['dep']['tail'])
        g_data[('token', '-dep', 'token')] = (rels['-dep']['head'], rels['-dep']['tail'])
        g_data[('token', 'loop', 'token')] = (rels['loop']['head'], rels['loop']['tail'])
        g = dgl.heterograph(g_data)
        return g


    def token2mask(self, sent1_tokens1, sent1_tokens2, sent2_tokens1, sent2_tokens2):
        mask = torch.zeros((len(sent1_tokens2)+len(sent2_tokens2), self.max_len))
        # sent1掩码
        end_index = 0
        for idx, word2 in enumerate(sent1_tokens2):
            for i, word1 in enumerate(sent1_tokens1):
                if word2.startswith(word1) and i >= end_index:
                    start_index = i  # 添加起始索引
                if word2.endswith(word1) and i >= end_index:
                    end_index = i  # 添加结束索引
                    break
            # 分词时首位添加[CLS]，索引需要+1
            start_index = start_index + 1
            end_index = end_index + 1
            # 判断起始位与终止位是否相同
            if start_index == end_index:
                current_index = [start_index]
            else:
                current_index = list(range(start_index, end_index + 1))
            mask[idx][current_index] = 1
        # sent2掩码
        end_index = 0
        for idx, word2 in enumerate(sent2_tokens2):
            for i, word1 in enumerate(sent2_tokens1):
                if word2.startswith(word1) and i + len(sent1_tokens1) + 2 >= end_index:
                    start_index = i + len(sent1_tokens1) + 2  # 添加起始索引
                if word2.endswith(word1) and i + len(sent1_tokens1) + 2 >= end_index:
                    end_index = i + len(sent1_tokens1) + 2  # 添加结束索引
                    break
            # 判断起始位与终止位是否相同
            if start_index == end_index:
                current_index = [start_index]
            else:
                current_index = list(range(start_index, end_index + 1))
            mask[idx+len(sent1_tokens2)][current_index] = 1
        return mask

    def __len__(self):
        return len(self.new_labels)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encoded["input_ids"][idx])
        token_type_ids = torch.tensor(self.encoded["token_type_ids"][idx])
        attention_mask = torch.tensor(self.encoded["attention_mask"][idx])
        graph = self.graph[idx]
        g_num_nodes = torch.tensor(self.g_num_nodes[idx])
        label = torch.tensor(self.new_labels[idx])
        return input_ids, token_type_ids, attention_mask, graph, g_num_nodes, label


