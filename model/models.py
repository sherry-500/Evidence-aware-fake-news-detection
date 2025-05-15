import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 

class MACAttention(nn.Module):
    def __init__(self, input_dim, output_dim, head_num=1):
        super().__init__()
        self.head_num = head_num
        self.W1 = nn.Linear(input_dim, output_dim, bias=False)
        self.w2 = nn.Linear(output_dim, head_num, bias=False)
        self.linear1 = nn.Linear(input_dim // 2, input_dim // 2 * head_num)
        self.linear2 = nn.Linear(input_dim // 2, input_dim // 2 * head_num)
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
        # c_hat = claim[:, :, 0, :].unsqueeze(2) # shape(batch_size, evi_num, 1, emb_dim)
        # e_hat = evidence[:, :, 0, :].unsqueeze(2) # shape(batch_size, evi_num, 1, emb_dim)
        c_hat = torch.mean(claim, dim=-2).unsqueeze(2) # shape(batch_size, evi_num, 1, emb_dim)
        e_hat = torch.mean(evidence, dim=-2).unsqueeze(2) # shape(batch_size, evi_num, 1, emb_dim)
        # claim = c_hat.expand(-1, -1, evidence.shape[2], -1) # shape(batch_size, evi_num, evidence_len, emb_dim)
        # attention = torch.tanh(self.W1(torch.cat([evidence, claim], dim=-1))) # shape(batch_size, evi_num, evidence_len, output_dim)
        # attention = self.w2(attention) # shape(batch_size, evi_num, evidence_len, head_num)
        # evidence_len_mask = evidence_len_mask.permute(0, 1, 3, 2).expand(-1, -1, -1, self.head_num) # shape(batch_size, evi_num, evidence_len, head_num)
        # attention_score = F.softmax(attention.masked_fill(~evidence_len_mask.type(torch.bool), -1e18), dim=-2)
        # output = attention_score.transpose(-2, -1) @ evidence # shape(batch_size, evi_num, head_num, emb_dim)
        # e_hat = output.flatten(start_dim=-2, end_dim=-1).unsqueeze(2) # shape(batch_size, evi_num, head_num * emb_dim)
        c_hat = self.linear1(c_hat)
        e_hat = self.linear2(e_hat)
        return c_hat, e_hat

class CoAttention(nn.Module):
    def __init__(self, input_dim, output_dim, head_num=1):
        super().__init__()
        self.head_num = head_num
        self.input_dim = input_dim
        self.W1 = nn.Linear(input_dim, output_dim, bias=False)
        self.w2 = nn.Linear(output_dim, head_num, bias=False)
        self.W2 = nn.Linear(input_dim, output_dim, bias=False)
        self.w1 = nn.Linear(output_dim, head_num, bias=False)
        # self.linear = nn.Linear(input_dim // 2, input_dim // 2 * head_num)

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
        claim1 = claim
        # factor = torch.sum(claim_len_mask, dim=-1)
        # factor[factor == 0] = 1
        # c_hat = (torch.sum(claim, dim=-2) / factor).unsqueeze(2)
        c_hat = torch.mean(claim, dim=-2).unsqueeze(2) # shape(batch_size, evi_num, 1, emb_dim)
        claim = c_hat.expand(-1, -1, evidence.shape[2], -1) # shape(batch_size, evi_num, evidence_len, emb_dim)
        attention = torch.tanh(self.W1(torch.cat([evidence, claim], dim=-1))) # shape(batch_size, evi_num, evidence_len, output_dim)
        attention = self.w2(attention) # shape(batch_size, evi_num, evidence_len, head_num)
        # attention = attention / np.sqrt(self.input_dim)
        evidence_len_mask = evidence_len_mask.permute(0, 1, 3, 2).expand(-1, -1, -1, self.head_num) # shape(batch_size, evi_num, evidence_len, head_num)
        attention_score1 = F.softmax(attention.masked_fill(~evidence_len_mask.type(torch.bool), -1e18), dim=-2) # shape(batch_size, evi_num, evidence_len, head_num)
        output = attention_score1.transpose(-2, -1) @ evidence # shape(batch_size, evi_num, head_num, emb_dim)
        e_hat = output.flatten(start_dim=-2, end_dim=-1).unsqueeze(2) # shape(batch_size, evi_num, head_num * emb_dim)

        # factor = torch.sum(evidence_len_mask.permute(0, 1, 3, 2), dim=-1)
        # factor[factor == 0] = 1
        # evidence = (torch.sum(evidence, dim=-2) / factor).unsqueeze(2)
        evidence = torch.mean(evidence, dim=-2).unsqueeze(2) # shape(batch_size, evi_num, 1, emb_dim)
        evidence = evidence.expand(-1, -1, claim1.shape[-2], -1) # shape(batch_size, evi_num, claim_len, emb_dim)
        attention = torch.tanh(self.W2(torch.cat([claim1, evidence], dim=-1)))
        attention = self.w1(attention) 
        # attention = attention / np.sqrt(self.input_dim)
        claim_len_mask = claim_len_mask.permute(0, 1, 3, 2).expand(-1, -1, -1, self.head_num)
        attention_score2 = F.softmax(attention.masked_fill(~claim_len_mask.type(torch.bool), -1e18), dim=-2) # shape(batch_size, evi_num, claim_len, head_num)
        output = attention_score2.transpose(-2, -1) @ claim1
        c_hat = output.flatten(start_dim=-2, end_dim=-1).unsqueeze(2)
        # c_hat = self.linear(c_hat) # shape(batch_size, evi_num, 1, emb_dim)
        
        return c_hat, e_hat, attention_s

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
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # self.dropout = nn.Dropout(0.2)
        # self.fc0 = nn.Linear(input_dim, input_dim * 2)
        # self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, input_dim // 48)
        # self.fc2 = nn.Linear(input_dim // 4, input_dim // 16)
        self.gelu = nn.GELU()
        self.fc3 = nn.Linear(input_dim // 48, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc3(out)
        return out

class Similarity(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # self.claim_len = claim_len
        # self.evi_len = evidence_len
        self.emb_dim = emb_dim
        self.weight_affinity = nn.Linear(emb_dim * 3, 1, bias=False)

    def forward(self, claim, evidence):
        """
        Arguments:
            claim: shape(batch_size, evi_num, emb_dim)
            evidence: shape(batch_size, evi_num, emb_dim)
        Returns:
            correlation: shape: (batch_size, evi_num)
        """
        # claim = self.layer_norm1(claim)
        # evidence = self.layer_norm1(evidence)
        # calculate the affinity between claim and evidence features 
        # at all pairs of claim-locations and evidence-locations
        ce = claim * evidence # shape(batch_size, evi_num, emb_dim)
        concatenated = torch.cat([claim, evidence, ce], dim=-1)
        affinity_mtx = torch.tanh(self.weight_affinity(concatenated)) # shape(batch_size, evi_num, 1)
        affinity_mtx = affinity_mtx.squeeze(-1) # shape: (batch_size, evi_num)
        
        return affinity_mtx

class FCModel(nn.Module):
    def __init__(self, hidden_dim, emb_dim, claim_src_num, evidence_src_num, evidence_src_dim, model_type=0, top_k=5, visualize=False):
        super().__init__()
        print(claim_src_num, evidence_src_num)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.claim_src_dim = 128
        self.evidence_src_dim = evidence_src_dim
        self.top_k = top_k
        # model_type
        # 0: 
        # 1: use claims' source
        # 2: use evidences' source
        # 3: use claim's source and evidences' source
        self.model_type = model_type
        self.visualize = visualize

        self.claim_src = nn.Embedding(claim_src_num, self.claim_src_dim)
        if model_type == 2 or model_type == 3:
            self.evidence_src = nn.Embedding(evidence_src_num, self.evidence_src_dim)
            print(self.evidence_src_dim)
        self.layer_norm = LayerNormalization(emb_dim)
        self.linear_c = nn.Linear(emb_dim, hidden_dim)
        self.linear_e = nn.Linear(emb_dim, hidden_dim)
        self.similarity = Similarity(hidden_dim * 1)
        self.relu = nn.ReLU()
        self.co_attention_w = CoAttention(hidden_dim, hidden_dim)
        # self.co_attention_w = MACAttention1(hidden_dim * 2, hidden_dim, 1)
        self.hierarchy_feature_combinator = HierarchyFeatureCombinator(self.emb_dim)
        # self.document_attention_layer = ConcatAttention(hidden_dim * 2, hidden_dim // 2)
        self.document_attention_layer = Attention()
        # self.document_attention_layer = CoAttention(hidden_dim, hidden_dim)
        input_dim = hidden_dim * 2 * 1
        if model_type == 1 or model_type == 3:
            input_dim += self.claim_src_dim
        if model_type == 2 or model_type == 3:
            input_dim += self.evidence_src_dim
        self.output_layer = MLP(input_dim, 1)

    def forward(self, claims, claims_source, claims_len_mask, evidences, evidences_len_mask, evidence_num_mask, evidences_source):
        """
        Arguments:
            claims: shape(batch_size, claim_len, emb_dim)
            claims_source: shape(batch_size)
            claims_len_mask: shape(batch_size, 1, claim_len)
            evidences: shape(batch_size, evi_num, evi_len, emb_dim)
            evidences_source: shape(batch_size, evi_num)
            evidences_len_mask: shape(batch_size, evi_num, 1, evidence_len)
        Returns:
            output: shape(batch_size)
        """
        evi_num = evidences.shape[1]
        
        claims = self.layer_norm(claims)
        evidences = self.layer_norm(evidences)
        
        claims = self.linear_c(claims)
        evidences = self.linear_e(evidences)

        outputs = self.co_attention_w(
            claims.unsqueeze(1).expand(-1, evi_num, -1, -1),
            claims_len_mask.unsqueeze(1).expand(-1, evi_num, -1, -1),
            evidences, evidences_len_mask
        )   
        c_w_hat = outputs[0] # shape: (batch_size, evi_num, 1, emb_dim)
        e_w_hat = outputs[1] # shape: (batch_size, evi_num, 1, emb_dim)
        
        if self.visualize:
            attention_map_e = outputs[2]  # shape(batch_size, evi_num, evidence_len, head_num)
            attention_map_c = outputs[3]  # shape(batch_size, evi_num, claim_len, head_num)
            
        # compute cosine similarity
        c_w_hat = c_w_hat.squeeze(2) # shape: (batch_size, evi_num, emb_dim)
        e_w_hat = e_w_hat.squeeze(2) # shape: (batch_size, evi_num, emb_dim)
        cosine_sim = self.similarity(c_w_hat, e_w_hat)
        cosine_sim = cosine_sim.masked_fill(~evidence_num_mask.type(torch.bool), -1)
        top_k_cosine_sim, top_k_indices = torch.topk(cosine_sim, self.top_k, dim=1, largest=True, sorted=True) # shape: (batch_size, top_k)
        top_k_indices_1 = top_k_indices.unsqueeze(2).repeat(1, 1, c_w_hat.shape[-1]) # shape: (batch_size, top_k, emb_dim)
        e_w_hat = torch.gather(e_w_hat, 1, top_k_indices_1) # shape: (batch_size, top_k, emb_dim)
        c_w_hat = torch.gather(c_w_hat, 1, top_k_indices_1) # shape: (batch_size, top_k, emb_dim)

        if self.model_type == 1 or self.model_type == 3:
            claims_source = self.claim_src(claims_source) # shape: (batch_size, claim_src_emb)
            claims_source = claims_source.unsqueeze(1).expand(-1, c_w_hat.shape[1], -1) # shape: (batch_size, top_k, claim_src_emb)
            c_w_hat = torch.cat([c_w_hat, claims_source], dim=-1)
        if self.model_type == 2 or self.model_type == 3:
            evidences_source = torch.gather(evidences_source, 1, top_k_indices)
            evidences_source = self.evidence_src(evidences_source) # shape: (batch_size, top_k, evidence_src_emb)
            e_w_hat = torch.cat([e_w_hat, evidences_source], dim=-1)

        # combine the features from the three level
        combined_feats = self.hierarchy_feature_combinator(c_w_hat, e_w_hat) # shape: (batch_size, top_k, emb_dim * 2)
        combined_feats = combined_feats.squeeze(2) # shape: (batch_size, top_k, 2 * emb_dim)

        # document level
        document_feats = self.document_attention_layer(top_k_cosine_sim, combined_feats) # shape: (batch_size, 1, emb_dim * 2)
        document_feats = document_feats.squeeze() # shape: (batch_size, emb_dim * 2)
        
        # classification layer
        output = self.output_layer(document_feats) # shape: (batch_size, 1)
        output = output.squeeze() # shape: (batch_size)

        if self.visualize:
            return cosine_sim, output, attention_map_c, attention_map_e
        else:
            return cosine_sim, output

    def set_visualize(self):
        self.visualize = True