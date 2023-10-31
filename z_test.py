import torch
from fairseq.models.bart import GuidedBARTModel

import sys
bart = GuidedBARTModel.from_pretrained(
    sys.argv[3],
    checkpoint_file=sys.argv[4],
    data_name_or_path=sys.argv[5]
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 100
beam=5
lenpen=1.0

with open(sys.argv[1]) as source,  open(sys.argv[2], 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=200, min_len=50, no_repeat_ngram_size=3)
                #hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
                #hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=999, min_len=5, no_repeat_ngram_size=3)
                #hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=999, min_len=5, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []
            zlines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=200, min_len=50, no_repeat_ngram_size=3)
        #hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        #hypotheses_batch = bart.sample(slines, beam=beam, lenpen=lenpen, max_len_b=999, min_len=5, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
