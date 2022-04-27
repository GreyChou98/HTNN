import torch



def get_span_mask(start_ids, end_ids, max_len):
    tmp = torch.arange(max_len).unsqueeze(0).expand(start_ids.shape[0], -1)
    batch_start_ids = start_ids.unsqueeze(1).expand_as(tmp)
    batch_end_ids = end_ids.unsqueeze(1).expand_as(tmp)
    if torch.cuda.is_available():
        tmp = tmp.cuda()
        batch_start_ids = batch_start_ids.cuda()
        batch_end_ids = batch_end_ids.cuda()
    mask = ((tmp >= batch_start_ids).float() * (tmp <= batch_end_ids).float()).unsqueeze(2)
    return mask