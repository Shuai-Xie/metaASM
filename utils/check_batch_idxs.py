from utils import load_json
from pprint import pprint

# batch0_uc_idxs = load_json('../exp/hope_cifar10_imb1_s0.4_r1.0_m50_Mar23_201207/batch0_uc_idxs.json')
epoch_total_uc_idxs = load_json('../exp/hope_cifar10_imb1_s0.4_r1.0_m100_Mar27_102601/epoch_total_uc_idxs.json')

print('epoch\tlen(idxs)\tcommon')
for epoch in range(30, 30 + len(epoch_total_uc_idxs) - 1):
    cur_ucs = epoch_total_uc_idxs[str(epoch)]
    next_ucs = epoch_total_uc_idxs[str(epoch + 1)]
    print(f'{epoch}-{epoch + 1}:\t{len(cur_ucs)}-{len(next_ucs)}\t{len(set(cur_ucs) & set(next_ucs))}')
