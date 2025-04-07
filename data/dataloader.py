import random
import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

class BucketSampler(Sampler):
    def __init__(self, data_source, bucket_size, shuffle=True):
        self.data_source = data_source
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        self.buckets = self._create_buckets()

    def _create_buckets(self):
        # Each example's format : (<claim>{torch.Tensor}, <claim_source>{torch.Tensor}, evidences{torch.Tensor}, evidences_source{torch.Tensor}, <cred_label>{int})
        indices = sorted(range(len(self.data_source)), key=lambda idx: (len(self.data_source[idx][0]) + len(self.data_source[idx][1])))
        buckets = [indices[i : i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        if self.shuffle:
            random.shuffle(buckets)
        return buckets

    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            yield from bucket
    
    def __len__(self):
        return len(self.data_source)

class BalancedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_idx_samples = self.data_source.class_idx_samples
        class_weight = self.data_source.class_weight
        assert 1 - sum(list(class_weight.values())) < 1e-6
        
        class_idx = list(class_weight.keys())
        num_in_batch = {i: math.floor(batch_size * class_weight[i]) for i in class_idx}
        _remain_num = batch_size - sum(num_in_batch.values())
        num_in_batch[random.choice(class_idx)] += _remain_num
        self.num_in_batch = num_in_batch
        self.offset_per_class = {i: 0 for i in class_idx}
        print(f'setting number_in_batch: {num_in_batch}')
        print('my sampler is inited.')

    def __iter__(self):
        # self.offset_per_class = {i: 0 for i in self.class_idx_samples.keys()}
        
        if self.shuffle:
            for c in self.class_idx_samples.keys():
                indices = torch.randperm(len(self.class_idx_samples[c]))
                self.class_idx_samples[c] = [self.class_idx_samples[c][i] for i in indices]
                
        batch = []
        i = 0
        while i < len(self):
            for c, num in self.num_in_batch.items():
                indices = self.data_source.class_idx_samples[c]
                start = self.offset_per_class[c]
                end = min(start + num, len(indices))
                batch += indices[start: end] + indices[0: (num - (end - start))]
                self.offset_per_class[c] = num - (end - start) if num - (end - start) > 0 else end
                if self.offset_per_class[c] == len(indices):
                    self.offset_per_class[c] = 0
            assert len(batch) == self.batch_size
            yield batch
            batch = []
            i += 1

    def __len__(self):
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

class collate_fn(object):
    def __init__(self, claim_src_vocab, evidence_src_vocab):
        # Hold this implementation specific arguments as the fields of the class.
        self.claim_src_vocab = claim_src_vocab
        self.evidence_src_vocab = evidence_src_vocab

    def __call__(self, batch):
        """
        Arguments:
            batch {list}: a batch of examples
        Returns:
            padded_claims {torch.Tensor}: shape(batch_size, seq_len, emb_dim)
            padded_claims_source {torch.Tensor}: shape(batch_size, seq_len, emb_dim)
            padded_evidences {torch.Tensor}: shape(batch_size, evi_num, seq_len, emb_dim)
            padded_evidences_source {torch.Tensor}: shape(batch_size, evi_num, seq_len, emb_dim)

            claims_len_mask {torch.Tensor}: shape(batch_size, 1, seq_len)
            claims_source_len_mask {torch.Tensor}: shape(batch_size, 1, seq_len)
            evidences_len_mask {torch.Tensor}: shape(batch_size, evi_num, 1, seq_len)
            evidences_source_len_mask {torch.Tensor}: shape(batch_size, evi_num, 1, seq_len)

            targets {torch.Tensor}: shape(batch_size, 1)
        """
        # Each example's format : (<claim>{torch.Tensor}, <claim_source>{torch.Tensor}, evidences{torch.Tensor}, evidences_source{torch.Tensor}, <cred_label>{int})
        claims, claims_source, evidences, evidences_source, targets = zip(*batch)
        
        targets = torch.tensor(targets)
        # padding to the same length
        padded_claims = pad_sequence(claims, batch_first=True)
        claims_source = [self.claim_src_vocab.get(src, 0) for src in claims_source]
        max_evi_len = max(tensor.shape[1] for tensor in evidences)
        max_evi_num = 5
        evidences = list(evidences)
        evidences_source = list(evidences_source)
        for i, e in enumerate(evidences):
            evidences[i] = F.pad(e, (0, 0, 0, max_evi_len - e.shape[1], 0, 0))
            max_evi_num = max(max_evi_num, evidences[i].shape[0])
        for i, s in enumerate(evidences_source):
            evidences_source[i] = [self.evidence_src_vocab.get(src, 0) for src in evidences_source[i]]

        # padding to the same evidence num 
        padded_evidences = pad_sequence(evidences, batch_first=True)
        padded_evidences = torch.stack([torch.cat((seqs, torch.zeros(max_evi_num - seqs.shape[0], * seqs.shape[1:], dtype=seqs.dtype)), dim=0) for seqs in padded_evidences])
        evidence_num = [len(evidence_source) for evidence_source in evidences_source]
        evidence_num = torch.tensor(evidence_num)
        evidence_num_mask = (torch.arange(max_evi_num).expand(len(evidences_source), max_evi_num) < evidence_num.unsqueeze(1)).int()
        evidences_source = [evidence_source + [0] * (max_evi_num - len(evidence_source)) for evidence_source in evidences_source]

        # reserve the length mask of original claim, claim_source, evidence, evidence_source
        claims_len_mask = (padded_claims != 0).any(dim=-1, keepdim=True).int().permute(0, 2, 1)
        evidences_len_mask = (padded_evidences != 0).any(dim=-1, keepdim=True).int().permute(0, 1, 3, 2)

        claims_source = torch.tensor(claims_source)
        evidences_source = torch.tensor(evidences_source)
        return (padded_claims, claims_source, claims_len_mask, padded_evidences, evidences_source, evidences_len_mask, evidence_num_mask), targets