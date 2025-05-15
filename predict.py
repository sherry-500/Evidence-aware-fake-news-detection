import shutil
import torch
import pickle
import json
from data.datasets import tok2emb_list, tok2emb_sent
from utils.visualize import render_bg_color
from transformers import BertModel, BertTokenizer

def predict(example, model_directory, bert_tokenizer, bert_model, cuda=True, visualize=True):
    """
    Arguments:
        example: string, json input in the format of string
        thr format of example: {
            'claim': ,
            'evidences': [
                {
                    'evidence': ,
                    'evidence_source': 
                },
                ...
            ],
            'claim_source': 
        }
    """
    with open(f'{model_directory}/claim_source_vocab.pickle', 'rb') as f:
        claim_src_vocab = pickle.load(f)
    with open(f'{model_directory}/evidence_source_vocab.pickle', 'rb') as f:
        evidence_src_vocab = pickle.load(f)
    if cuda:
        model = torch.load(f'{model_directory}/checkpoint.pt')
        bert_model.to('cuda')
        model.to('cuda')
    else:
        model = torch.load(f'{model_directory}/checkpoint.pt', map_location=torch.device('cpu'))
    if visualize:
        model.set_visualize()
    model.eval()
    
    try:
        data = json.loads(example)
    except json.JSONDecodeError:
        print("Error: Invalid JSON input")
        return

    claim = tok2emb_sent(data['claim'], bert_tokenizer, bert_model, cuda)
    claim_source = data['claim_source']
    claim_src_idx = claim_src_vocab.get(claim_source, 0)

    evidences = []
    evidences_source_idx = []
    for evidence in data['evidences']:
        evidences.append(evidence['evidence'])
        evidence_source = evidence['evidence_source']
        evidences_source_idx.append(evidence_src_vocab.get(evidence_source, 0))

    claim_token = bert_tokenizer.tokenize(data['claim'])
    evidences_token = [bert_tokenizer.tokenize(evidence) for evidence in evidences]
    
    evidences = tok2emb_list(evidences, bert_tokenizer, bert_model, cuda)
    evi_num = 5 if len(evidences) < 5 else len(evidences)
    evidence_num = evidences.shape[0]
    evidences = torch.cat((evidences, torch.zeros(evi_num - evidences.shape[0], * evidences.shape[1:], dtype=evidences.dtype)), dim=0)
    evidence_num_mask = (torch.arange(evi_num) < evidence_num).int()
    
    claim_len_mask = (claim != 0).any(dim=-1, keepdim=True).int().permute(1, 0)
    evidences_len_mask = (evidences != 0).any(dim=-1, keepdim=True).int().permute(0, 2, 1)
    
    claim_source = torch.tensor(claim_src_idx)
    evidences_source = evidences_source_idx + [0] * (evi_num - len(evidences_source_idx))
    evidences_source = torch.tensor(evidences_source)

    inputs = (claim, claim_source, claim_len_mask, evidences, evidences_len_mask, evidence_num_mask, evidences_source)
    inputs = (param.unsqueeze(0) for param in inputs)
    if cuda:
        inputs = (param.to('cuda') for param in inputs)

    if visualize:
        _, label, attention_c, attention_e = model(*inputs)
        attention_c_list = attention_c.squeeze(0)[:, 1:, :].tolist() # shape: (evi_num, claim_len, head_num)
        attention_e_list = attention_e.squeeze(0)[:, 1:, :].tolist() # shape: (evi_num, evidence_len, head_num)
        
        terminal_width = shutil.get_terminal_size().columns
        underline = '_' * (terminal_width + 2)

        for i, (attention_c, attention_e) in enumerate(zip(attention_c_list, attention_e_list)):
            if i >= len(data['evidences']):
                break
            print(underline)
            # print_color_text(claim_token, attention_c)
            print(f"\033[1mClaim:[{data['cred_label']}]\033[0m", end=' ')
            render_bg_color(claim_token, attention_c)
            print()
            print(f'\033[1mRelevant Article {i}:\033[0m', end=' ')
            # print_color_text(evidences_token[i], attention_e)                    
            render_bg_color(evidences_token[i], attention_e)
            print()
        print(underline)
    else:
        _, label = model(*inputs)
        
    label = label.to('cpu') > 0
    print()
    print(f'\033[1mPrediction: [{label.item()}]\033[0m')

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    with open('test_4.jsonl', 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            if "actor christopher walken planning making bid us presidency 2008" in data['claim']:
                # data = json.loads(line.strip())
                target = 1 if data['cred_label'] == 'True' else 0
                predict(line, 'checkpoint/PolitiFact/0', tokenizer, bert_model, cuda=False)
                break