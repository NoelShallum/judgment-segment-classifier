"""Extract only named FP32 state tensors from the pinned public LegalSeg archive."""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import pickle
import zipfile
import numpy as np
from safetensors.numpy import save_file

REV = '837c2a85ce9a6df21c9e3ef6d1b908b23b66d985'
BASE = f'https://huggingface.co/L-NLProc/LegalSeg_Hier_BiLSTM-CRF/resolve/{REV}/Hier_BiLSTM-CRF/'
SOURCE_SHA = '2f3096bd97db2ae5addce8d9b8fc3d750072e5a224c6b05ba2a52759a0cb36a2'

class Placeholder:
    def __new__(cls, *args):
        return object.__new__(cls)
    def __setstate__(self, state):
        pass

@dataclass
class FloatView:
    storage: tuple
    offset: int
    size: tuple
    stride: tuple

def view(storage, offset, size, stride, *ignored):
    return FloatView(storage, offset, size, stride)

class StateReader(pickle.Unpickler):
    def persistent_load(self, token):
        name, dtype, index, device, count = token
        if name != 'storage' or dtype is not Placeholder or not index.isdigit() or count < 0:
            raise ValueError('Unexpected stored array descriptor')
        return index, count
    def find_class(self, module, name):
        if (module, name) == ('collections','OrderedDict'):
            return OrderedDict
        if (module, name) == ('torch._utils','_rebuild_tensor_v2'):
            return view
        if (module, name) == ('torch._utils','_rebuild_parameter'):
            return lambda item, *args: item
        if (module, name) == ('__builtin__','set'):
            return set
        if (module, name) == ('_codecs','encode'):
            return lambda *args: b''
        ignored = {'model.Hier_BiLSTM_CRF':{'Hier_LSTM_CRF_Classifier'},'model.submodels':{'LSTM_Sentence_Encoder','LSTM_Emitter','CRF'},'torch.nn.modules.dropout':{'Dropout'},'torch.nn.modules.linear':{'Linear'},'torch.nn.modules.rnn':{'LSTM'},'torch.nn.modules.sparse':{'Embedding'},'numpy':{'dtype'},'numpy.core.multiarray':{'scalar'},'torch':{'FloatStorage'}}
        if name in ignored.get(module, set()):
            return Placeholder
        raise ValueError(f'Unexpected serialized global {module}.{name}')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def extract(src, dst):
    if sha(src) != SOURCE_SHA:
        raise ValueError('Unexpected source SHA-256')
    with zipfile.ZipFile(src) as archive:
        prefix = 'model_state4/'
        if archive.read(prefix+'byteorder') != b'little':
            raise ValueError('Unexpected byte order')
        state = StateReader(io.BytesIO(archive.read(prefix+'data.pkl'))).load()['state_dict']
        weights = {}
        for key, descriptor in state.items():
            if not isinstance(descriptor, FloatView):
                raise ValueError(f'Unexpected state tensor {key}')
            index, count = descriptor.storage
            raw = np.frombuffer(archive.read(prefix+'data/'+index), dtype='<f4')
            end = descriptor.offset + int(np.prod(descriptor.size))
            if len(raw) != count or descriptor.offset < 0 or end > count:
                raise ValueError(f'Invalid storage shape for {key}')
            arr = raw[descriptor.offset:end].reshape(descriptor.size)
            if tuple(i//4 for i in arr.strides) != descriptor.stride:
                raise ValueError(f'Noncontiguous storage for {key}')
            weights[key] = arr
    save_file(weights, str(dst))
    return {k:list(v.shape) for k,v in weights.items()}

if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('archive',type=Path); ap.add_argument('output',type=Path); args=ap.parse_args()
    print(extract(args.archive,args.output))
