# Med-BERT Plan For Fixed-Length Patient Disease-History Embeddings

## Goal

Use the Med-BERT architecture and repo workflow to produce one fixed-length latent vector per patient from structured disease history in AoU-style OMOP data.

Implement and compare two methods:

1. Method 1: use the full disease history directly and produce one patient vector from all codes across all visits.
2. Method 2: first produce one vector per visit, then aggregate visit vectors into one patient vector.

Both methods must produce one vector of the same length for every patient.

## Short Answer On Model Availability

- Med-BERT code is public, but the original pretrained checkpoint is no longer shared by the authors.
- BEHRT code is public, but a public pretrained checkpoint is not obvious from the repo.
- ClinicalBERT checkpoints do exist publicly, but they are for clinical note text, not structured diagnosis-code sequences.
- Therefore, if the goal is truly Med-BERT-style structured disease-history embeddings, plan for local Med-BERT pretraining unless you already have a compatible checkpoint.

## Workspace-Specific Context

The local notebooks already suggest the right AoU source tables:

- `querying_patients.ipynb` uses `condition_occurrence` and joins concept/vocabulary metadata.
- `ck_join_starter.ipynb` shows fields including:
  - `person_id`
  - `visit_occurrence_id`
  - `condition_start_datetime`
  - `condition_source_value`
  - `source_vocabulary`
  - `condition_concept_id`

This is enough to build Med-BERT-style patient visit sequences.

## Key Constraint

Med-BERT was designed for structured diagnosis histories and expects tokenized diagnosis codes arranged as chronologically ordered visits. The original config uses `max_position_embeddings = 512`, so long patient histories must be chunked. Do not rely on a single forward pass for all patients.

## Recommended Overall Strategy

1. Extract diagnosis histories from AoU OMOP.
2. Normalize them into Med-BERT-style tokens.
3. Build patient visit sequences.
4. If no checkpoint exists, pretrain a local Med-BERT checkpoint on AoU disease histories.
5. Convert the checkpoint to PyTorch if needed.
6. Build an embedding extraction script that uses `last_hidden_state`, not the default first-token pooler.
7. Implement both patient-vector methods.
8. Save patient vectors and run sanity checks.

## Important Med-BERT Facts To Preserve

From the paper and repo:

- Input is a sequence of visits, each visit a list of diagnosis codes.
- Med-BERT uses:
  - code embeddings
  - serialization embeddings for within-visit order
  - visit embeddings for visit identity/order
- The paper does not rely on `[CLS]` as the patient summary.
- The repo/paper treat sequence representation as a pooling/summing problem over code outputs.

Implication:

- For embeddings, extract `last_hidden_state` for all real tokens/codes.
- Do not use `pooler_output` as the patient vector.

## Data Mapping From AoU To Med-BERT Input

### Source tables

Use:

- `condition_occurrence`
- `visit_occurrence`
- `concept`
- `vocabulary`

### Minimum fields to extract

For each diagnosis row, extract:

- `person_id`
- `visit_occurrence_id`
- `condition_start_date`
- `condition_start_datetime`
- `visit_start_date`
- `visit_end_date`
- `condition_source_value`
- `source_vocabulary`
- `condition_concept_id`
- `condition_type_concept_id`

### Which diagnosis vocabulary to use

Recommended:

- keep only `ICD9CM` and `ICD10CM`

Reason:

- Med-BERT was pretrained mainly on ICD-9 and ICD-10 diagnosis codes.
- This keeps the token space close to the original Med-BERT design.

### Build Med-BERT-style diagnosis tokens

Convert each diagnosis row to a token string:

- if `source_vocabulary == "ICD9CM"`, token = `ICD9_<normalized_code>`
- if `source_vocabulary == "ICD10CM"`, token = `ICD10_<normalized_code>`

Normalization rules:

1. uppercase the code
2. strip surrounding whitespace
3. keep decimal points if present
4. remove internal spaces
5. drop rows with null or empty codes

Examples:

- `585.3` with `ICD9CM` -> `ICD9_585.3`
- `N18.3` with `ICD10CM` -> `ICD10_N18.3`

## Visit Construction Rules

Group rows by:

- `person_id`
- `visit_occurrence_id`

For each visit:

1. sort rows by `condition_start_datetime`
2. break ties by `condition_type_concept_id`
3. break remaining ties by `condition_concept_id`
4. break final ties by token string

Then:

- deduplicate repeated identical tokens within the same visit
- keep the first occurrence after sorting

### If diagnosis priority / POA are unavailable

The Med-BERT preprocessing script sorts by fields like POA and diagnosis priority in Cerner. AoU may not expose those in the same form.

Use this fallback:

- within-visit order = deterministic sorted order from the rules above
- serialization index = `1, 2, 3, ...` within each visit

This is acceptable because Med-BERT can handle ordered or unordered codes inside a visit.

## Output Formats To Build

Create two dataset forms.

### Format A: Med-BERT pretraining TSV

Build a tab-delimited file with columns matching the repo’s expectation:

- `patient_id`
- `vadate`
- `vddate`
- `diag`
- `poa`
- `diagnosis_priority`
- `third_party_ind`

AoU mapping:

- `patient_id` <- `person_id`
- `vadate` <- `visit_start_date`
- `vddate` <- `visit_end_date`
- `diag` <- normalized Med-BERT token
- `poa` <- placeholder `0` if unavailable
- `diagnosis_priority` <- within-visit running index `1..n`
- `third_party_ind` <- placeholder `0`

Important:

- the Med-BERT preprocessing code mainly uses these fields for grouping, sorting, and vocabulary creation
- dummy values are acceptable for unavailable fields as long as order is deterministic

### Format B: Direct Python sequence objects for inference

After preprocessing, create a Python object per patient containing:

- `person_id`
- ordered list of visits
- each visit containing ordered tokens
- visit dates
- optional metadata:
  - total code count
  - total visit count
  - first visit date
  - last visit date

This second format is easier for chunking and embedding extraction.

## Branching Logic

### Branch 1: no Med-BERT checkpoint exists

Do this:

1. create AoU Med-BERT TSV
2. run Med-BERT preprocessing
3. create TFRecord features
4. pretrain Med-BERT
5. convert checkpoint to PyTorch
6. run embedding extraction

### Branch 2: a compatible Med-BERT checkpoint already exists

Skip the pretraining stage and start at embedding extraction.

## Pretraining Plan

### Step 1: preprocess raw AoU TSV with Med-BERT repo script

Use the repo’s preprocessing path:

1. `preprocess_pretrain_data.py`
2. `create_BERTpretrain_EHRfeatures.py`
3. `run_EHRpretraining.py` or `run_EHRpretraining_QA2Seq.py`

### Step 2: create vocabulary

Use `preprocess_pretrain_data.py` with `vocab = NA` on the first full build so it creates the `.types` vocabulary file.

Save:

- token -> integer mapping
- reverse integer -> token mapping

### Step 3: quality filters

Before pretraining, apply these filters:

- drop patients with zero valid visits
- drop patients with zero valid codes
- recommended: keep only patients with at least 4 diagnosis codes total
- recommended: optionally keep only patients with at least 2 visits if long-range context is important

### Step 4: split strategy

For self-supervised pretraining:

- split patients, not rows
- recommended split:
  - 70% train
  - 10% valid
  - 20% test

### Step 5: train Med-BERT

Start with the original config unless compute constraints force a smaller model.

Original repo config:

- `hidden_size = 192`
- `num_hidden_layers = 6`
- `num_attention_heads = 6`
- `max_position_embeddings = 512`

Recommendation:

- keep `hidden_size = 192` initially
- that makes every final patient vector length `192`

### Step 6: convert checkpoint for easier inference

After training, convert TF checkpoint to PyTorch.

Reason:

- the repo’s downstream examples are easier to adapt in PyTorch
- vector extraction is simpler using `BertModel` outputs in PyTorch

## Embedding Extraction Primitives

Implement these functions.

### `normalize_icd_code(row) -> str | None`

Input:

- `condition_source_value`
- `source_vocabulary`

Output:

- normalized token like `ICD10_N18.3`
- `None` if unusable

### `build_patient_visits(df) -> dict[person_id, patient_record]`

Output per patient:

- ordered visits
- each visit has:
  - `visit_occurrence_id`
  - `visit_start_date`
  - `visit_end_date`
  - ordered `token_ids`
  - ordered `token_strings`

### `chunk_patient_sequence(patient_record, max_seq_len=512) -> list[chunks]`

Rules:

1. preserve chronological order
2. try not to split a visit across chunks
3. if adding a full visit would exceed 512 codes, emit the current chunk and start a new one
4. if a single visit itself has more than 512 codes, split that visit into subchunks and mark them as belonging to the same original visit

Each chunk must contain:

- `input_ids`
- `attention_mask`
- `segment_ids`
- mapping from token positions back to original visit ids

### `encode_chunk(model, chunk) -> tensor`

Return:

- `last_hidden_state` of shape `[seq_len, hidden_size]`

Only keep rows where `attention_mask == 1`.

## Method 1: Full-History Direct Patient Vector

### Idea

Use all diagnosis codes across all visits, encode them with Med-BERT, and directly pool all code embeddings into one patient vector.

### Why this matches Med-BERT

- it uses the full contextual sequence
- it avoids the misleading first-token pooler
- it is closest to the paper’s “sum outputs across codes” idea

### Implementation steps

1. For each patient, flatten all visits into one chronological token stream.
2. Assign visit segment ids so all codes from the same visit share the same visit id.
3. Chunk if total codes > 512.
4. For each chunk:
   - run Med-BERT
   - collect `last_hidden_state` for real tokens only
5. Concatenate all token embeddings from all chunks in chronological order.
6. Compute one patient vector by masked mean pooling over all token embeddings.

### Exact pooling rule

If token embeddings for a patient are `H` with shape `[N_tokens, hidden_size]`:

- `patient_vector = mean(H, axis=0)`

Optional ablation:

- also save `max(H, axis=0)` for comparison
- but use mean as the primary method

### Why choose mean instead of sum

- mean keeps vector scale stable across patients with different history length
- every patient gets the same vector length
- fewer confounding effects from utilization volume

### Handling long histories

If a patient spans multiple chunks:

1. compute token embeddings chunk by chunk
2. append all real token embeddings
3. mean-pool across the combined token list

This preserves “all visits” without truncation.

### Final output

For each patient save:

- `person_id`
- `embedding_method = "method1_full_history_mean"`
- `embedding_dim = hidden_size`
- `embedding` as a float array

## Method 2: Per-Visit Vectors Then Aggregate

### Idea

First reduce each visit to one visit vector, then aggregate the visit vectors into one patient vector.

### Why this is worth trying

- separates within-visit code composition from across-visit disease trajectory
- makes visit-level inspection possible
- easier to add temporal weighting

### Step A: create one vector per visit

For each patient chunk:

1. run Med-BERT to get token embeddings
2. use the token-to-visit mapping to group token embeddings by original visit
3. for each visit, mean-pool its token embeddings

If a visit was split across chunks:

1. gather all token embeddings belonging to that visit from all chunks
2. mean-pool across all of them once

Result:

- one visit vector per original visit
- shape per visit = `[hidden_size]`

### Step B: aggregate visit vectors into one patient vector

Use a deterministic recency-weighted mean as the primary aggregator.

#### Recency-weighted aggregation rule

For patient visit vectors `V_1 ... V_T` ordered oldest to newest:

1. compute days from each visit to the patient’s last visit:
   - `delta_t = last_visit_date - visit_date`
2. convert to weights:
   - `w_t = exp(-lambda * delta_t_in_days)`
3. normalize weights so they sum to 1
4. compute:
   - `patient_vector = sum_t (w_t * V_t)`

Recommended starting value:

- `lambda = 0.001`

This gives more weight to recent visits without discarding early history.

### Fallback if dates are unreliable

Use unweighted mean over visit vectors:

- `patient_vector = mean(V_1 ... V_T)`

### Optional learned variant

Only do this after the deterministic version works.

Replace the recency-weighted mean with:

- a 1-layer GRU over visit vectors, then use the final hidden state
- or a small attention pooling layer over visit vectors

Do not make this the first implementation because it adds training complexity.

### Final outputs

Save two artifacts:

1. visit-level table
   - `person_id`
   - `visit_occurrence_id`
   - `visit_index`
   - `visit_date`
   - `visit_embedding`
2. patient-level table
   - `person_id`
   - `embedding_method = "method2_visit_recency_weighted_mean"`
   - `embedding_dim = hidden_size`
   - `embedding`

## Recommended Comparison Protocol

For each patient, produce:

- Method 1 vector
- Method 2 vector

Then compare them using:

1. vector norm distribution
2. missingness
3. cosine similarity distribution
4. nearest-neighbor clinical face validity
5. simple downstream test performance

## Simple Downstream Sanity Test

Use a small labeled task already relevant to this repo, such as CKD case/control.

For each method:

1. generate one vector per patient
2. split by patient
3. train a simple downstream model:
   - logistic regression
   - random forest
4. compare:
   - AUC
   - AUCPR
   - calibration
5. prefer the method that is both stable and predictive

Reason:

- this is the fastest way to tell whether the embedding is clinically useful

## File Layout Recommendation

Use something like:

- `data/raw/aou_conditions.parquet`
- `data/intermediate/medbert_pretrain.tsv`
- `data/intermediate/medbert_vocab.pkl`
- `data/intermediate/patient_visits.pkl`
- `models/medbert_tf_checkpoint/...`
- `models/medbert_pytorch/...`
- `outputs/embeddings/method1_patient_embeddings.parquet`
- `outputs/embeddings/method2_visit_embeddings.parquet`
- `outputs/embeddings/method2_patient_embeddings.parquet`
- `outputs/qc/embedding_qc_report.json`

## Concrete Coding Task Breakdown

A less capable coding agent should implement the work in this order.

1. Write SQL to extract AoU diagnosis rows from `condition_occurrence` and `visit_occurrence`.
2. Write a normalization function that maps AoU ICD rows to `ICD9_*` or `ICD10_*` tokens.
3. Write a patient-visit builder that groups by `person_id` and `visit_occurrence_id`.
4. Write a Med-BERT TSV exporter with placeholder `poa` and `third_party_ind`.
5. Run Med-BERT preprocessing to create:
   - `.types`
   - `.bencs.train`
   - `.bencs.valid`
   - `.bencs.test`
6. Create TFRecord features with `create_BERTpretrain_EHRfeatures.py`.
7. Pretrain Med-BERT if no checkpoint exists.
8. Convert the checkpoint to PyTorch.
9. Write an inference wrapper that returns `last_hidden_state`.
10. Write a chunker that preserves chronology and records token-to-visit mapping.
11. Implement Method 1 pooling.
12. Implement Method 2 visit pooling and recency-weighted aggregation.
13. Save embeddings.
14. Run QC checks.
15. Run one downstream sanity benchmark.

## Quality Checks

Run all of these.

### Data QC

- number of patients extracted
- number of visits per patient
- number of valid ICD codes per patient
- fraction of rows dropped for invalid vocabulary/code
- vocabulary size
- top 100 most frequent tokens

### Sequence QC

- max codes per patient
- max codes per visit
- fraction of patients requiring chunking
- number of visits split across chunks

### Embedding QC

- all vectors have identical length
- no NaNs
- no all-zero vectors
- mean and variance per dimension look reasonable
- nearest neighbors are not random noise
- embedding norms are not almost perfectly driven by code count

## Main Risks And Mitigations

### Risk 1: no public Med-BERT checkpoint

Mitigation:

- plan for local pretraining
- if compute is limited, pretrain a smaller Med-BERT first and verify the pipeline end to end

### Risk 2: AoU lacks Cerner-specific fields like POA or diagnosis priority

Mitigation:

- use deterministic within-visit ordering
- fill unavailable fields with placeholders for compatibility

### Risk 3: long patient histories exceed 512 tokens

Mitigation:

- chunk by visit
- aggregate chunk outputs instead of truncating

### Risk 4: pooler output is misleading

Mitigation:

- always use `last_hidden_state`
- do not use first-token pooling as the patient vector

### Risk 5: Method 2 aggregator is too heuristic

Mitigation:

- start with deterministic recency-weighted mean
- later compare against mean and GRU-based aggregation

## Recommendation

Use this as the default path:

1. train a local Med-BERT checkpoint on AoU ICD-9/10 histories
2. extract token-level hidden states
3. implement Method 1 as masked mean over all token embeddings across all visits and chunks
4. implement Method 2 as:
   - per-visit mean pooling
   - recency-weighted mean over visit vectors
5. compare both with a simple downstream CKD prediction benchmark

## Optional Fallback If You Want Public Weights Immediately

If you need a public checkpoint before structured Med-BERT is working:

1. convert each patient’s diagnosis history to text using diagnosis descriptions
2. encode the text with `emilyalsentzer/Bio_ClinicalBERT`
3. pool text embeddings into one vector per patient

But this is a fallback, not a true Med-BERT replacement.

## Open Questions

1. Should the AoU extraction keep only `ICD9CM` and `ICD10CM`, or also map SNOMED-only rows into text/code aliases?
2. Do you want the first implementation to include local Med-BERT pretraining, or should it first prototype the end-to-end embedding pipeline on a small subset?
3. Should the downstream sanity benchmark use the CKD cohort already implied by the local notebooks, or a different phenotype?
