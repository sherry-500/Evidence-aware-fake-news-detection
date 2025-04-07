import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 

class CoAttention(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        # self.claim_len = claim_len
        # self.evi_len = evidence_len
        self.emb_dim = emb_dim
        self.layer_norm1 = LayerNormalization(emb_dim, requires_grad=False)
        self.layer_norm2 = LayerNormalization(hidden_dim, requires_grad=False)
        # Parameters for computing the affinity matrix
        self.weight_affinity = nn.Parameter(
            init.xavier_normal_(torch.empty(emb_dim, emb_dim), gain=1.0),
            requires_grad=True
        )
        # Parameters for attention computation
        self.weight_c = nn.Parameter(
            init.xavier_normal_(torch.empty(emb_dim, hidden_dim)),
            requires_grad=True
        )
        self.weight_e = nn.Parameter(
            init.xavier_normal_(torch.empty(emb_dim, hidden_dim)),
            requires_grad=True
        )
        self.weight_hc = nn.Parameter(
            torch.randn(hidden_dim, 1),
            requires_grad=True
        )
        self.weight_he = nn.Parameter(
            torch.randn(hidden_dim, 1),
            requires_grad=True
        )

    def forward(self, claim, claim_len_mask, evidence, evidence_len_mask):
        """
        Arguments:
            claim: shape(batch_size, evi_num, claim_len, emb_dim)
            claim_len_mask: shape(batch_size, evi_num, 1, claim_len)
            evidence: shape(batch_size, evi_num, evidence_len, emb_dim)
            evidence_len_mask: shape(batch_size, evi_num, 1, evidence_len)
        Returns:
            c_hat: shape: (batch_size, evi_num, 1, emb_dim)
            e_hat: shape: (batch_size, evi_num, 1, emb_dim)
        """
        claim = self.layer_norm1(claim)
        evidence = self.layer_norm1(evidence)
        # calculate the affinity between claim and evidence features 
        # at all pairs of claim-locations and evidence-locations
        # shape: (batch_size, evi_num, claim_len, evidence_len)
        affinity_mtx = torch.tanh(claim @ self.weight_affinity @ evidence.permute(0, 1, 3, 2))
        # affinity_mtx = claim @ self.weight_affinity @ evidence.permute(0, 1, 3, 2)
        
        # calculate the attention-based claim and evidence representations
        # shape: (batch_size, evi_num, claim_len, hidden_dim)
        H_c = torch.tanh(claim @ self.weight_c + affinity_mtx @ evidence @ self.weight_e)
        # shape(batch_size, evi_num, evidence_len, hidden_dim)
        H_e = torch.tanh(evidence @ self.weight_e + affinity_mtx.permute(0, 1, 3, 2) @ claim @ self.weight_c)
        
        # the claim attention map
        # shape: (batch_size, evi_num, 1, claim_len)
        attention_c = self.weight_hc.T @ H_c.permute(0, 1, 3, 2)
        masked_a_c = F.softmax(attention_c.masked_fill(~claim_len_mask.type(torch.bool), -1e18), dim=-1)
        # the evidence attention map
        # shape: (batch_size, evi_num, 1, evidence_len)
        attention_e = self.weight_he.T @ H_e.permute(0, 1, 3, 2)
        masked_a_e = F.softmax(attention_e.masked_fill(~evidence_len_mask.type(torch.bool), -1e18), dim=-1)

        # weighted sum of claim and evidence features
        # shape: (batch_size, evi_num, 1, emb_dim)
        c_hat = (masked_a_c @ claim)
        # shape: (batch_size, evi_num, 1, emb_dim)
        e_hat = (masked_a_e @ evidence)

        return c_hat, e_hat

class LayerNormalization(nn.Module):
    def __init__(self, feats_num, epsilon=1e-6, requires_grad=True):
        super().__init__()
        self.epsilon = epsilon
        self.gain = nn.Parameter(torch.ones(feats_num), requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(feats_num), requires_grad=requires_grad)

    def forward(self, x):
        """
        Arguments:
            x {torch.Tensor}: shape(batch_size, seq_len, feats_num)
        Returns:
            {torch.Tensor}: shape(batch_size, seq_len, feats_num)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gain * (x - mean) / (std + self.epsilon) + self.bias 

class HierarchyFeatureCombinator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, c_w, e_w):
        """
        Arguments:
            c_w: shape(batch_size, evi_num, 1, input_dim)
            e_w: shape(batch_size, evi_num, 1, input_dim)
            c_p: shape(batch_size, evi_num, 1, input_dim)
            e_p: shape(batch_size, evi_num, 1, input_dim)
            c_a: shape(batch_size, evi_num, 1, input_dim)
            e_a: shape(batch_size, evi_num, 1, input_dim)
        Returns:
            feat : # shape: (batch_size, evi_num, 1, input_dim * 2)
        """
        feat1 = torch.cat([c_w, e_w], dim=-1) # shape: (batch_size, evi_num, 1, input_dim * 2)
        # feat2 = torch.cat([c_p, e_p], dim=-1) # shape: (batch_size, evi_num, 1, input_dim * 2)
        # feat3 = torch.cat([c_a, e_a], dim=-1) # shape: (batch_size, evi_num, 1, input_dim * 2)
        # feat = feat1 + feat2 # shape: (batch_size, evi_num, 1, input_dim * 2)
        return feat1

class ConcatAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.layer_norm = LayerNormalization(hidden_dim, requires_grad=False)
        self.W = nn.Parameter(init.xavier_normal_(torch.empty(input_dim, hidden_dim)), requires_grad=True)
        self.v_a = nn.Parameter(torch.randn(1, hidden_dim), requires_grad=True)

    def forward(self, x):
        """
        Arguments:
            x shape(batch_size, evi_num, input_dim): the concatenated matrix
        Returns:
            x shape(batch_size, 1, input_dim)
        """
        attention_score = self.v_a @ self.tanh(self.layer_norm((x @ self.W)).transpose(1, 2)) # shape: (batch_size, 1, evi_num)
        attention = F.softmax(attention_score, dim=-1) # shape: (batch_size, 1, evi_num)
        x = attention @ x # shape: (batch_size, 1, input_dim)
        return x

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cosine_sim, x):
        """
        Arguments:
            cosine_sim shape(batch_size, evi_num)
            x shape(batch_size, evi_num, input_dim): the concatenated matrix
        Returns:
            x shape(batch_size, input_dim)
        """
        attention = F.softmax(cosine_sim, dim=-1).unsqueeze(1) # shape: (batch_size, 1, evi_num)
        x = attention @ x # shape: (batch_size, 1, input_dim)
        x = torch.cat([x, cosine_sim.unsqueeze(1)], dim=-1)
        # x = torch.sum(x, dim=1)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, input_dim // 12)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.gelu = nn.GELU()
        self.fc3 = nn.Linear(input_dim // 12, output_dim)

    def forward(self, x):
        # out = self.dropout(x)
        out = self.fc1(x)
        # out = self.gelu(out)
        # out = self.fc2(out)
        # out = self.gelu(out)
        out = self.fc3(out)

        return out

class FCModel(nn.Module):
    def __init__(self, hidden_dim, emb_dim, claim_src_num, evidence_src_num, top_k=5):
        super().__init__()
        print(claim_src_num, evidence_src_num)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.claim_src_dim = 10
        self.evidence_src_dim = 20
        self.top_k = top_k

        self.claim_src = nn.Embedding(claim_src_num, self.claim_src_dim)
        self.evidence_src = nn.Embedding(evidence_src_num, self.evidence_src_dim)
        self.layer_norm = LayerNormalization(emb_dim)
        self.linear_c = nn.Linear(emb_dim, hidden_dim)
        self.linear_e = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.co_attention_w = CoAttention(hidden_dim, hidden_dim)
        # self.phrase_extractor_c = PhraseFeatureExtractor(self.emb_dim, self.emb_dim // 16, filter_dims=[3, 5, 6])
        # self.phrase_extractor_e = PhraseFeatureExtractor(self.emb_dim, self.emb_dim // 16, filter_dims=[3, 5, 6])
        # self.article_encoder = TransformerEncoder(layer_num=1, head_num=1, feats_num=self.emb_dim // 2, hidden_dim=self.hidden_dim, dropout_prob=0.3)
        self.hierarchy_feature_combinator = HierarchyFeatureCombinator(self.emb_dim)
        # self.document_attention_layer = ConcatAttention(hidden_dim * 2, hidden_dim)
        self.document_attention_layer = Attention()
        # self.output_layer = nn.Linear(self.emb_dim * 2, 1)
        self.output_layer = MLP(hidden_dim * 2 + top_k + self.claim_src_dim, 1)

    def forward(self, claims, claims_source, claims_len_mask, evidences, evidences_source, evidences_len_mask, evidence_num_mask):
        """
        Arguments:
            claims: shape(batch_size, claim_len, emb_dim)
            claims_source: shape(batch_size, claim_source_len, emb_dim)
            claims_len_mask: shape(batch_size, 1, claim_len)
            evidences: shape(batch_size, evi_num, evi_len, emb_dim)
            evidences_source: shape(batch_size, evi_num, evi_source_len, emb_dim)
            evidences_len_mask: shape(batch_size, evi_num, 1, evidence_len)
        Returns:
            output: shape(batch_size)
        """
        evi_num = evidences.shape[1]
        # split_evidence = torch.unbind(evidences, dim=1)
        # split_evi_len_mask = torch.unbind(evidences_len_mask, dim=1)

        claims = self.layer_norm(claims)
        evidences = self.layer_norm(evidences)
        
        claims = self.linear_c(claims)
        evidences = self.linear_e(evidences)
        
        c_w_hat, e_w_hat = self.co_attention_w(
            claims.unsqueeze(1).expand(-1, evi_num, -1, -1),
            claims_len_mask.unsqueeze(1).expand(-1, evi_num, -1, -1),
            evidences, evidences_len_mask
        )   # shape: (batch_size, evi_num, 1, emb_dim)

        c_w_hat, e_w_hat = self.co_attention_w(
            claims.unsqueeze(1).expand(-1, evi_num, -1, -1),
            claims_len_mask.unsqueeze(1).expand(-1, evi_num, -1, -1),
            evidences, evidences_len_mask
        )   # shape: (batch_size, evi_num, 1, emb_dim)

        # compute cosine similarity
        c_w_hat = c_w_hat.squeeze(2) # shape: (batch_size, evi_num, emb_dim)
        e_w_hat = e_w_hat.squeeze(2) # shape: (batch_size, evi_num, emb_dim)
        cosine_sim = F.cosine_similarity(c_w_hat, e_w_hat, dim=-1) # shape: (batch_size, evi_num)
        cosine_sim = cosine_sim.masked_fill(~evidence_num_mask.type(torch.bool), -2)
        top_k_cosine_sim, top_k_indices = torch.topk(cosine_sim, self.top_k, dim=1, largest=True, sorted=True) # shape: (batch_size, top_k)
        top_k_indices_1 = top_k_indices.unsqueeze(2).repeat(1, 1, self.hidden_dim) # shape: (batch_size, top_k, emb_dim)
        e_w_hat = torch.gather(e_w_hat, 1, top_k_indices_1) # shape: (batch_size, top_k, emb_dim)
        c_w_hat = torch.gather(c_w_hat, 1, top_k_indices_1) # shape: (batch_size, top_k, emb_dim)

        # extend claim and evidence with their source
        claims_source = self.claim_src(claims_source) # shape: (batch_size, claim_src_emb)
        claims_source = claims_source.unsqueeze(1).expand(-1, self.top_k, -1) # shape: (batch_size, top_k, claim_src_emb)
        # evidences_source = torch.gather(evidences_source, 1, top_k_indices) # shape: (batch_size, top_k)
        # evidences_source = self.evidence_src(evidences_source) # shape: (batch_size, top_k, evidence_src_emb)
        c_w_hat = torch.cat([c_w_hat, claims_source], dim=-1)
        # e_w_hat = torch.cat([e_w_hat, evidences_source], dim=-1)

        # combine the features from the three level
        combined_feats = self.hierarchy_feature_combinator(c_w_hat, e_w_hat) # shape: (batch_size, top_k, emb_dim * 2)
        combined_feats = combined_feats.squeeze(2) # shape: (batch_size, top_k, 2 * emb_dim)

        # document level
        document_feats = self.document_attention_layer(top_k_cosine_sim, combined_feats) # shape: (batch_size, 1, emb_dim * 2)
        document_feats = document_feats.squeeze() # shape: (batch_size, emb_dim * 2)
        
        # classification layer
        output = self.output_layer(document_feats) # shape: (batch_size, 1)
        output = output.squeeze() # shape: (batch_size)

        return output