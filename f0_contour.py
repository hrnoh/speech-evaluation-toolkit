import os
import glob
import numpy as np
import pickle
import tqdm
import re
from utils import get_logf0, load_wav, speaker_norm

data_root = "C:\\Users\\hrnoh\\OneDrive - 고려대학교\\LAB\\내연구\\음성\\MOS\\"
out_file = "f0_contour_sr22050.pkl"
M = ['p226', 'p227', 'p232', 'p302', 'p304', 'p311']
F = ['p225', 'p228', 'p229', 'p230', 'p231', 'p233', 'p234', 'p301', 'p303', 'p305', 'p306', 'p307', 'p308', 'p310']

models = ['StarGAN-VC', 'AutoVC', 'Seq2seq-VC', 'DCVC']
GT_path = os.path.join(data_root, 'GT')

out_dict = {}

# make GT
all_wav_path = glob.glob(os.path.join(data_root, models[0], '*.wav'))
logf0_dict = {
    'm2m': [],
    'm2f': [],
    'f2m': [],
    'f2f': []
}
print(" [*] {} start!".format('GT'))
for wav_path in tqdm.tqdm(all_wav_path):
    wav_name = os.path.basename(wav_path)

    pattern = r"p[0-9]+_[0-9]+"
    src, trg = re.findall(pattern, wav_name)

    wav_path_gt = os.path.join(data_root, 'GT', trg + '.wav')
    wav_gt = load_wav(wav_path_gt, 22050)
    logf0_gt = get_logf0(wav_gt, 22050, frame_period=(256 / (0.001 * 22050)))
    logf0_gt = speaker_norm(logf0_gt)

    src_spk = src.split('_')[0]
    trg_spk = trg.split('_')[0]

    each_dict = {'GT': logf0_gt}
    for m in models:
        temp = os.path.join(data_root, m, wav_name)

        wav = load_wav(temp, 22050)
        logf0 = get_logf0(wav, 22050, frame_period=(256 / (0.001 * 22050)))
        logf0 = speaker_norm(logf0)
        each_dict[m] = logf0

    if src_spk in M and trg_spk in M:
        logf0_dict['m2m'].append(each_dict)
    elif src_spk in M and trg_spk in F:
        logf0_dict['m2f'].append(each_dict)
    elif src_spk in F and trg_spk in M:
        logf0_dict['f2m'].append(each_dict)
    elif src_spk in F and trg_spk in F:
        logf0_dict['f2f'].append(each_dict)

with open(out_file, 'wb') as f:
    pickle.dump(logf0_dict, f)