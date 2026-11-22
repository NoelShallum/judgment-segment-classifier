"""Materialize the validation selection and donor into a runnable tensor patch."""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from run_role_edge import HERE, OUT, file_hash, save_json


def main():
    freeze=json.loads((HERE/'frozen.json').read_text())
    training=json.loads((HERE/'train.json').read_text())
    source=HERE/'role_edge.safetensors'
    assert file_hash(source)==freeze['weights_sha256']
    prior=load_file(str(source))
    manifest={'selection_sha256':file_hash(HERE/'frozen.json'),'vocab_sha256':training['vocab_sha256'],
              'base_sha256':file_hash(HERE.parent/'references/hier/hier.safetensors'),'profiles':{}}
    for name,(mode,gain) in freeze['choices'].items():
        current=OUT/f'whole-discourse-{name}.safetensors'
        assert file_hash(current)==freeze['parents_sha256'][name]
        out=HERE/f'role-edge-{name}.safetensors'
        packed={**load_file(str(current)),**prior,'edge.mode':np.array(int(mode=='interaction'),np.int32),
                'edge.gain':np.array(gain,np.float32)}
        save_file(packed,str(out))
        read=load_file(str(out))
        assert all(np.array_equal(read[k],v) for k,v in packed.items())
        manifest['profiles'][name]={'file':out.name,'sha256':file_hash(out),'mode':mode,'gain':float(gain),
                                  'tensors':{k:list(v.shape) for k,v in packed.items()}}
    save_json(HERE/'deployment.json',manifest)

if __name__=='__main__':main()
