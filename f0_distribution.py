import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm
import re
from utils import get_logf0, load_wav

data_root = "C:\\Users\\cvpr\\OneDrive - 고려대학교\\LAB\\내연구\\음성\\MOS\\"
out_file = "f0_dist.pkl"
M = ['p226', 'p227', 'p232', 'p302', 'p304', 'p311']
F = ['p225', 'p228', 'p229', 'p230', 'p231', 'p233', 'p234', 'p301', 'p303', 'p305', 'p306', 'p307', 'p308', 'p310']

models = ['StarGAN-VC', 'AutoVC', 'Seq2seq-VC', 'DCVC']
GT_path = os.path.join(data_root, 'GT')

out_dict = {}

# make GT
all_wav_path = glob.glob(os.path.join(data_root, models[0], '*.wav'))
logf0_dict = {}
m2m_gt = []
m2f_gt = []
f2m_gt = []
f2f_gt = []
print(" [*] {} start!".format('GT'))
for wav_path in tqdm.tqdm(all_wav_path):
    wav_name = os.path.basename(wav_path)

    pattern = r"p[0-9]+_[0-9]+"
    src, trg = re.findall(pattern, wav_name)

    wav_path_gt = os.path.join(data_root, 'GT', trg + '.wav')

    src_spk = src.split('_')[0]
    trg_spk = trg.split('_')[0]

    wav = load_wav(wav_path_gt, 22050)
    logf0 = get_logf0(wav, 22050, frame_period=(256 / (0.001 * 22050)))

    if src_spk in M and trg_spk in M:
        m2m_gt.append(logf0[logf0 > 0])
    elif src_spk in M and trg_spk in F:
        m2f_gt.append(logf0[logf0 > 0])
    elif src_spk in F and trg_spk in M:
        f2m_gt.append(logf0[logf0 > 0])
    elif src_spk in F and trg_spk in F:
        f2f_gt.append(logf0[logf0 > 0])

logf0_dict['m2m'] = np.concatenate(m2m_gt)
logf0_dict['m2f'] = np.concatenate(m2f_gt)
logf0_dict['f2m'] = np.concatenate(f2m_gt)
logf0_dict['f2f'] = np.concatenate(f2f_gt)
out_dict['GT'] = logf0_dict

for model in models:
    all_wav_path = glob.glob(os.path.join(data_root, model, '*.wav'))

    logf0_dict = {}
    m2m = []
    m2f = []
    f2m = []
    f2f = []

    print(" [*] {} start!".format(model))
    for wav_path in tqdm.tqdm(all_wav_path):
        wav_name = os.path.basename(wav_path)
        wav = load_wav(wav_path, sr=22050)
        logf0 = get_logf0(wav, 22050, frame_period=(256 / (0.001 * 22050)))

        pattern = r"p[0-9]+"
        src, trg = re.findall(pattern, wav_name)

        if src in M and trg in M:
            m2m.append(logf0[logf0 > 0])
        elif src in M and trg in F:
            m2f.append(logf0[logf0 > 0])
        elif src in F and trg in M:
            f2m.append(logf0[logf0 > 0])
        elif src in F and trg in F:
            f2f.append(logf0[logf0 > 0])

    logf0_dict['m2m'] = np.concatenate(m2m)
    logf0_dict['m2f'] = np.concatenate(m2f)
    logf0_dict['f2m'] = np.concatenate(f2m)
    logf0_dict['f2f'] = np.concatenate(f2f)
    out_dict[model] = logf0_dict

with open(out_file, 'wb') as f:
    pickle.dump(out_dict, f)