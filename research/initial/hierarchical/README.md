# Seven-role CPU model for Indian judgments

This directory is complete. With Python 3.11, install torch, numpy and safetensors. The first command returns one dominant role for either a sentence or a paragraph. The next two return half-open Unicode character ranges covering every input character.

```sh
python hier_model.py 'We therefore allow the appeal.'
python hier_model.py --spans --text-file whole-judgment.txt
python hier_model.py --fragment --text-file excerpt-from-middle.txt
```

The 16,816,110-parameter donor weighs 67,266,352 bytes in FP32 plus 3,175,220 vocabulary bytes. The excerpt profile removes two sets of seven learned boundary weights. The complete-judgment profile halves all 49 role transitions. The interior-passage profile has its own validation-selected weights. Exact counts and SHA-256 digests are in profile.json. Supplying --unedited runs the original tensors with the same zero initial states. The publisher's original code samples random recurrent states. Punctuation-only units receive an unknown token to preserve offsets. See the included NOTICE and the evaluation brief.

For source reconstruction, download model_state4.tar from the source revision in profile.json and run `python hier_extract.py model_state4.tar reconstructed.safetensors`. The extractor checks the upstream SHA-256 and reads tensor storage through an allowlisted metadata reader.
