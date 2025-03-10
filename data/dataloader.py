import random
import torch
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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

def collate_fn(batch):
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
    padded_claims_source = pad_sequence(claims_source, batch_first=True)
    max_evi_len = max(tensor.shape[1] for tensor in evidences)
    max_evi_src_len = max(tensor.shape[1] for tensor in evidences_source)
    max_evi_num = 5
    evidences = list(evidences)
    evidences_source = list(evidences_source)
    for i, e in enumerate(evidences):
        evidences[i] = F.pad(e, (0, 0, 0, max_evi_len - e.shape[1], 0, 0))
        max_evi_num = max(max_evi_num, evidences[i].shape[0])
    for i, s in enumerate(evidences_source):
        evidences_source[i] = F.pad(s, (0, 0, 0, max_evi_src_len - s.shape[1], 0, 0))

    # padding to the same evidence num 
    padded_evidences = pad_sequence(evidences, batch_first=True)
    padded_evidences = torch.stack([torch.cat((seqs, torch.zeros(max_evi_num - seqs.shape[0], * seqs.shape[1:], dtype=seqs.dtype)), dim=0) for seqs in padded_evidences])
    padded_evidences_source = pad_sequence(evidences_source, batch_first=True)
    padded_evidences_source = torch.stack([torch.cat((seqs, torch.zeros(max_evi_num - seqs.shape[0], * seqs.shape[1:], dtype=seqs.dtype)), dim=0) for seqs in padded_evidences_source])
    
    # reserve the length mask of original claim, claim_source, evidence, evidence_source
    claims_len_mask = (padded_claims != 0).any(dim=-1, keepdim=True).int().permute(0, 2, 1)
    claims_source_len_mask = (padded_claims_source != 0).any(dim=-1, keepdim=True).int().permute(0, 2, 1)
    evidences_len_mask = (padded_evidences != 0).any(dim=-1, keepdim=True).int().permute(0, 1, 3, 2)
    evidences_source_len_mask = (padded_evidences_source != 0).any(dim=-1, keepdim=True).int().permute(0, 1, 3, 2)

    return (padded_claims, padded_claims_source, claims_len_mask, claims_source_len_mask, 
            padded_evidences, padded_evidences_source, evidences_len_mask, evidences_source_len_mask), targets