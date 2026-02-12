# MolLangData GitHub — TODO

This file tracks the planned work for this repository. The repo is **under construction**.

---

## 1. Full pipeline to collect the data

### 1.1 Simple generation code

- [x] Minimal script: input = IUPAC (and SMILES) + metadata (e.g. from OPSIN XML).
- [x] Output = one structure description via a single LLM call.
- [x] Reuse prompt logic from `get_prompt_description_from_iupac.py` (e.g. `prompts/smiles_iupac_metadata_v13/`).

### 1.2 Routing example

- [x] Document or script showing **difficulty-based routing** (easy / medium / hard).
- [x] Map to model and reasoning effort (e.g. high vs xhigh) as in the dataset stats.
- [x] Optional: semantic routing (e.g. bridged / fused / spiro) for prompt variant selection.

### 1.3 Whole pipeline (batch API)

- [ ] Prepare prompts from a molecule list (IUPAC or SMILES + OPSIN for XML where needed).
- [ ] Call batch API to generate descriptions (e.g. OpenAI-style batch).
- [ ] Optional: validation (e.g. atom count, pass@k) and export to Parquet / HF dataset format.

---

## 2. OPSIN

- [ ] **README and usage** are maintained in the fork: [feiyang-cai/opsin_mollangdata](https://github.com/feiyang-cai/opsin_mollangdata).
- [ ] This repo: document how the compiled JAR is used in our pipeline and link to the OPSIN fork.

---

## 3. Integration

- [x] **Demo** (IUPAC → XML + prompt) is in repo root: `get_prompt_description_from_iupac.py`, JAR, and `prompts/`.
- [ ] Align `requirements.txt` and any scripts with the HF dataset repo (`mollangdata_hf`) where relevant.
