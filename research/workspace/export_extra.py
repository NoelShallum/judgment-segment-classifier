"""Include a CPU-ready hierarchy, source/checksum notice and optional assembled readouts."""
from pathlib import Path
import json
from package_candidate import install,write
from seven_paths import ROOT,LABELS
from weight_edit import file_hash

def build(kit,out):
    donor=ROOT/'references/hier';profiles=ROOT/'checkpoints/hierarchical'
    d=kit/'hierarchical';ready=out/'hierarchical-model'
    source='https://huggingface.co/L-NLProc/LegalSeg_Hier_BiLSTM-CRF/tree/837c2a85ce9a6df21c9e3ef6d1b908b23b66d985/Hier_BiLSTM-CRF'
    for p in [donor/'hier.safetensors',donor/'word2idx.json',profiles/'excerpt.safetensors',profiles/'document.safetensors',profiles/'fragment.safetensors',ROOT/'hier_model.py',ROOT/'span_model.py',ROOT/'hier_extract.py',kit/'LICENSE']:
        if p.exists():install(p,d/p.name)
    m=json.loads((ROOT/'results/hierarchical-validation.json').read_text())
    info={'revision':m['revision'],'parent_archive_sha256':m['donor_sha256'],'source':source,'parameters':m['parameters'],'labels':LABELS,'files':{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in d.iterdir() if p.suffix in {'.safetensors'} or p.name=='word2idx.json'}}
    write(d/'profile.json',json.dumps(info,indent=2)+'\n')
    write(d/'NOTICE','LegalSeg hierarchical BiLSTM-CRF donor and vocabulary by Nigam, Dubey, Sharma, Shallum, Ghosh and Bhattacharya. Donor release tagged Apache License 2.0. AttoFlow reserialized the tensors, uses deterministic recurrent states, and directly programmed alternative emission and transition profiles.\n'+source+'\nhttps://aclanthology.org/2025.findings-naacl.63/\n')
    write(d/'README.md','''# Seven-role CPU model for Indian judgments

This directory is complete. With Python 3.11, install torch, numpy and safetensors. The first command returns one dominant role for either a sentence or a paragraph. The next two return half-open Unicode character ranges covering every input character.

```sh
python hier_model.py 'We therefore allow the appeal.'
python hier_model.py --spans --text-file whole-judgment.txt
python hier_model.py --fragment --text-file excerpt-from-middle.txt
```

The 16,816,110-parameter donor weighs 67,266,352 bytes in FP32 plus 3,175,220 vocabulary bytes. The excerpt profile removes two sets of seven learned boundary weights. The complete-judgment profile halves all 49 role transitions. The interior-passage profile has its own validation-selected weights. Exact counts and SHA-256 digests are in profile.json. Supplying --unedited runs the original tensors with the same zero initial states. The publisher's original code samples random recurrent states. Punctuation-only units receive an unknown token to preserve offsets. See the included NOTICE and the evaluation brief.

For source reconstruction, download model_state4.tar from the source revision in profile.json and run `python hier_extract.py model_state4.tar reconstructed.safetensors`. The extractor checks the upstream SHA-256 and reads tensor storage through an allowlisted metadata reader.
''')
    for p in d.iterdir():
        if p.is_file():install(p,ready/p.name)
    for obsolete in ('prepare_fresh.py','prepare_matched.py','prepare_confirmation.py'):
        (kit/'reproduce'/obsolete).unlink(missing_ok=True)
    base=(kit/'README.md').read_text()
    add='''## Run the smaller model offline

The `hierarchical` directory includes the complete 16.82M-parameter recurrent model and vocabulary. From the extracted archive run:

```sh
python hierarchical/hier_model.py 'We therefore allow the appeal.'
python hierarchical/hier_model.py --spans --text-file whole-judgment.txt
python hierarchical/hier_model.py --fragment --text-file excerpt-from-middle.txt
```

The complete-judgment and clipped-passage programs were selected on separate development samples. Their exact-span F1 values are task-specific. The BERT profiles below reconstruct a second model family.

'''
    write(kit/'README.md',base.replace('## Run on a CPU',add+'## Run the BERT profiles on a CPU'))
    fusion=ROOT/'results/fusion-validation.json'
    if fusion.is_file():
        report=json.loads(fusion.read_text());spec=report['excerpt'];d=kit/'combined'
        for p in (ROOT/'predict_combined.py',ROOT/'hier_model.py',ROOT/'span_model.py',ROOT/'windowing.py'):install(p,d/p.name)
        install(ROOT/'checkpoints/fusion/excerpt.safetensors',d/'combined.safetensors')
        write(d/'README.md','Reconstruct the BERT '+('document' if spec['bert']=='b' else 'compact')+' profile with rebuild.py as shown in the archive README. In the archive root run `python combined/predict_combined.py --bert-dir document-model --hier-dir hierarchical "We therefore dismiss the appeal."` (the case shown uses donor B). The physically written matrix combines 768 BERT and 200 recurrent features. The development-selected document experiment assigned zero coefficient to BERT and is already served by hierarchical/hier_model.py --spans.\n')
    for sub in ('mlp_states','mlp_write','fusion_trial','hier_trial','hier_full','hier_extract','hier_model','hier_verify','fragment_trial','role_cues','cue_trial','gguf_choice_rows','prepare_projection','score_projection','semantic_projection_prep','projection_follow','source_filter','stocktake_alignment','prepare_span_cloud','span_cloud_metrics','span_eval','predict_combined'):
        p=ROOT/(sub+'.py')
        if p.is_file():install(p,kit/'reproduce'/p.name)
