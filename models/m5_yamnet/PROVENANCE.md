# Official YAMNet model provenance

- Model: Google YAMNet, TensorFlow 2 variation, version 1.
- Official model listing: <https://tfhub.dev/google/yamnet/1> (redirects to Google's Kaggle Models publisher page; model card lists Apache-2.0).
- Official Keras checkpoint: <https://storage.googleapis.com/audioset/yamnet.h5>.
- Checkpoint object size: 15,296,092 bytes.
- Google Cloud Storage `x-goog-hash` MD5: `8234225c045859752bd4a2e101f8cd1a` (base64 response value `gjQiXARYWXUr1KLhAfjNGg==`).
- Verified checkpoint MD5: `8234225c045859752bd4a2e101f8cd1a`.
- Verified checkpoint SHA-256: `13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744`.
- Official TensorFlow Models source revision: [`c14bf9ad91962cf189f9f58db2132c06247fcd53`](https://github.com/tensorflow/models/commit/c14bf9ad91962cf189f9f58db2132c06247fcd53), signature reported verified by GitHub's official API.
- The vendored inference modules and repository license are under `src/biosonic_yamnet/_official/`; source SHA-256 values are recorded in `source_provenance.json`.

## Licensing and attribution

The Google YAMNet model listing identifies the pretrained model as Apache License 2.0. TensorFlow Models source files are also covered by Apache-2.0 in the `research` tree; the original upstream license text is retained beside the copied source. Preserve upstream notices and the license when redistributing those source files. The direct Google Storage checkpoint URL does not provide a separate license file alongside the binary, so the checkpoint license attribution here follows the official model listing; do not infer additional rights from the source repository license alone.

YAMNet was trained on AudioSet. AudioSet's dataset is CC BY 4.0 and its class ontology is CC BY-SA 4.0 under [AudioSet's terms](https://research.google.com/audioset/download.html). The vendored `yamnet_class_map.csv` contains AudioSet class indices, MIDs, and display names; attribute Google AudioSet and retain the ontology's CC BY-SA 4.0 terms when redistributing it. These dataset/ontology terms are distinct from the model listing's Apache-2.0 license. BioSonic does not redistribute AudioSet audio. The class map is not BioSonic-authored taxonomy.

This provenance applies only to this checkpoint and its pinned upstream source. It does not determine the license for BirdCLEF data or for BioSonic as a project.
