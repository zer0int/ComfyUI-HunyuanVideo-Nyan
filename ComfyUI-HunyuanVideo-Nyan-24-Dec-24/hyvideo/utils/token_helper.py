import torch
import torch.nn.functional as F

def find_subsequence(sequence, sub_sequence):

    assert sequence.shape[0]==1
    sequence = sequence[0]
    sub_sequence = sub_sequence[0]

    sub_len = len(sub_sequence)
    indices = []
        
    windows = sequence.unfold(0, sub_len, 1)
    matches = (windows == sub_sequence).all(dim=1)
    indices = matches.nonzero().flatten().tolist()

    return indices, len(indices), sub_len

import ast
import torch

def multi_slice_to_mask(expr, length):
    def process_single_slice(s):
        s = s.replace(':', ',').replace(' ', '')
        while ',,' in s:
            s = s.replace(',,', ',None,')
        if s.startswith(','):
            s = 'None' + s
        if s.endswith(','):
            s = s + 'None'
        return s
    
    try:
        slices = expr.split(',')
        mask = torch.zeros(length, dtype=torch.bool)
        if expr == "":
            return mask
        i = 0
        while i < len(slices):
            if ':' in slices[i]:
                slice_expr = process_single_slice(slices[i])
                slice_args = ast.literal_eval(f"({slice_expr})")
                s = slice(*slice_args)
                mask[s] = True
                i += 1
            else:
                idx = ast.literal_eval(slices[i])
                if idx < 0:
                    idx = length + idx
                if 0 <= idx < length:
                    mask[idx] = True
                i += 1
                
        return mask
    except Exception as e:
        raise ValueError(f"Invalid slice expression: {e}")
