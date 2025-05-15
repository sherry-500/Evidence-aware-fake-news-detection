import numpy as np

ansi_code_ids =  [15, 224, 217, 210, 203, 160, 124, 88]
template = "\033[38;5;{value}m{string}\033[0m"

ansi_bg_ids = ['255;255;255', '255;240;240', '255;218;218', '255;142;142', '255;100;100', '255;77;77']
template_bg = "\033[48;2;{value}m{string}\033[0m"

def render_bg_color(text, ws):
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for string, w in zip(text, ws):
        value = ansi_bg_ids[int(w.item() * len(ansi_bg_ids))]
        print(template_bg.format(string=string, value=value), end=" ")
    
def print_color_text(text, ws):
    '''
    Arguments:
        text: list
        ws: list
    '''
    ws = np.array(ws)
    ws = (ws - np.min(ws)) / (np.max(ws) - np.min(ws)) * 0.99
    for string, w in zip(text, ws):
        vid = int(w * len(ansi_code_ids))
        value = ansi_code_ids[vid]
        print(template.format(string=string, value=value), end=" ")