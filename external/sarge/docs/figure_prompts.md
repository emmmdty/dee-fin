# SARGE CCKS 2026 Figure Prompts

Local project root: `/home/tjk/myProjects/masterProjects/DEE/SARGE/`
Server project root: `/data/TJK/DEE/SARGE/`
Local Python: `/home/tjk/miniconda3/envs/feg-dev-py310/bin/python`
Server Python: `/data/TJK/envs/sarge_vllm_full/bin/python`

These prompts document the first-version PNG figure generation process for the CCKS 2026 LNCS paper. Generated
image bases are not trusted for dense text. Final figure text is overlaid or redrawn with deterministic
TikZ/PDF/PIL rendering so labels remain readable and exact.

## Fig. 1: Binding Gap

- Figure ID: `fig1_binding_gap_v1`
- Intended message: Correct role values do not imply correct event records; homogeneous events share public
  context and repeated anchors, while record-specific anchors determine separation.
- Source/reference files:
  - `paper/ccks_2026/figures/fig1_binding_gap.png`
  - `paper/ccks_2026/figures/drafts/fig1_binding_gap_imagegen_draft.png`
  - `paper/ccks_2026/figures/ppt/fig1_binding_gap_reference_fully_editable.pptx`
  - `paper/ccks_2026/figures/reference/fig1_binding_gap_previous_vector_preview.png`
- Required visual elements:
  - Left panel: simplified Chinese financial announcement with three equity pledge entries.
  - Middle panel: shared context, record-specific/repeated anchors, and visible binding-gap warning.
  - Right panel: fixed-slot records R1/R2/R3.
  - Bottom takeaway: `Role values != Record assembly` or `Document-level association matters`.
- Exact labels to appear:
  - `Shared context / 共享上下文`
  - `Record-specific / repeated anchors`
  - `Binding Gap / 绑定缺口`
  - `Schema-valid records / 模式有效记录`
  - `Role values != Record assembly`
- Style guide: clean academic conference diagram, flat vector-like, high contrast, minimal decoration, no
  photorealism, no 3D, no glossy effects.
- Color palette:
  - Navy `#1F4E79`
  - Teal `#2A9D8F`
  - Amber `#E9C46A`
  - Coral `#E76F51`
  - Slate `#6C757D`
  - Light background `#F8FAFC`
  - Border gray `#CBD5E1`
  - Warning red `#DC2626`
  - Success green `#16A34A`
- Layout/aspect ratio: 16:9 landscape, target PNG width at least 3200 px.
- Image generation prompt:

```text
Create a clean vector-style academic conference diagram for Chinese financial document-level event extraction.
Three left-to-right panels. Panel 1: a simplified Chinese financial announcement with three highlighted equity
pledge entries, each containing a shareholder, pledgee, share amount, and registration date. Panel 2:
homogeneous event ambiguity: shared context shown in blue chips, record-specific and repeated anchors
including pledgee, share amount, and start date shown in teal/coral chips, and a red warning box labeled
'Binding Gap / 绑定缺口'. Panel 3: schema-valid
fixed-slot records R1, R2, R3 with rows for Company, Pledgee, Shares, StartDate. Use flat vector graphics,
rounded rectangles, thin arrows, navy section headers, light blue background, blue for shared context, teal and
coral for record anchors, coral/red for binding errors. Academic LNCS paper style, high readability, no photorealism, no
3D, no decorative icons. Leave enough blank space for exact text overlay. 16:9, high resolution.
```

- Negative prompt:

```text
no photorealistic people, no logos, no stock-market icons, no clutter, no illegible tiny text, no random Chinese
characters, no distorted tables, no watermark, no dark background, no gradient-heavy poster style
```

- Post-processing plan for exact text overlay: keep the final manuscript asset at
  `paper/ccks_2026/figures/fig1_binding_gap.png`. Keep image-generation drafts under `figures/drafts/` and
  editable/reference assets under `figures/ppt/` or `figures/reference/`.

## Fig. 2: SARGE Pipeline

- Figure ID: `fig2_sarge_pipeline_v1`
- Intended message: SARGE separates auxiliary evidence exposure, role-grounded SFT, deterministic generation,
  validation, and canonical export.
- Source/reference files:
  - `paper/ccks_2026/figures/fig2_sarge_pipeline.png`
  - `paper/ccks_2026/figures/drafts/fig2_sarge_pipeline_imagegen_draft.png`
  - `paper/ccks_2026/figures/ppt/fig2_sarge_pipeline_reference_fully_editable.pptx`
  - `paper/ccks_2026/figures/reference/fig2_sarge_pipeline_previous_vector_preview.png`
- Required visual elements:
  - Input financial announcement.
  - Dataset schema.
  - Optional dashed modules: Surface Memory candidates and Slot Plan prior.
  - Role-safe prompt contract with minimal readable labels, not dense JSON.
  - Qwen3-4B + LoRA SFT.
  - Greedy decoding.
  - Strict JSON parser.
  - Schema validation.
  - Conservative anchor export.
  - Canonical fixed-slot records.
  - Diagnostic readout: `100% schema-valid, 0 parse failures` and `Binding bottleneck remains`.
- Exact labels to appear:
  - `SARGE: Schema-Grounded Role-Aware Generation`
  - `Input financial announcement`
  - `Dataset schema`
  - `Surface Memory candidates`
  - `Slot Plan prior`
  - `Role-safe prompt contract`
  - `Qwen3-4B + LoRA SFT`
  - `Greedy decoding`
  - `Strict JSON parser`
  - `Schema validation`
  - `Conservative anchor export`
  - `Canonical fixed-slot records`
  - `100% schema-valid, 0 parse failures`
  - `Binding bottleneck remains`
- Style guide: clean academic pipeline diagram, rounded rectangles, thin arrows, dashed optional boxes, solid
  core boxes, no decorative icons.
- Color palette: same as Fig. 1; use green for validity diagnostics and coral/red for binding bottleneck.
- Layout/aspect ratio: 16:9 or slightly wider, target PNG width at least 3200 px.
- Image generation prompt:

```text
Create a clean vector-style academic pipeline diagram titled 'SARGE: Schema-Grounded Role-Aware Generation'.
Left-to-right workflow for Chinese financial document-level event extraction. Inputs on the left: financial
announcement and dataset schema. Optional dashed modules above: Surface Memory candidates and Slot Plan prior.
Center: Role-safe prompt contract with only a few readable labels feeding Qwen3-4B + LoRA SFT, then greedy decoding. Right: strict JSON
parser, schema validation, conservative anchor export, canonical fixed-slot records. Bottom diagnostic strip:
green badge '100% schema-valid, 0 parse failures' and coral badge 'Binding bottleneck remains'. Use consistent
academic conference visual style, rounded rectangles, thin arrows, navy headers, light background, teal/blue for
schema and prompt, amber for optional evidence, green for valid output, coral/red for bottleneck. High
readability, minimal text, no tiny JSON or table snippets, no decorative icons. 16:9, high resolution.
```

- Negative prompt:

```text
no photorealistic server racks, no robots, no random code text, no tiny unreadable JSON, no distorted labels, no
watermark, no dark theme, no excessive gradients, no 3D
```

- Post-processing plan for exact text overlay: keep the final manuscript asset at
  `paper/ccks_2026/figures/fig2_sarge_pipeline.png`. Keep image-generation drafts under `figures/drafts/` and
  editable/reference assets under `figures/ppt/` or `figures/reference/`.

## Optional Fig. 3: Exact-Record Buckets

- Figure ID: `fig3_exact_record_buckets_v1`
- Intended message: Role-level F1 remains much higher than Exact-Record as same-type record count increases.
- Source/reference files:
  - `runs/sarge_record_diagnostics_seed13_20260525/eval/record_count_buckets.csv`
  - `runs/sarge_record_diagnostics_seed13_20260525/summary.json`
  - `paper/ccks_2026/figures/fig3_exact_record_buckets.png`
  - `paper/ccks_2026/figures/ppt/sarge_final_png_figures.pptx`
- Required visual elements: compact grouped bar or point chart for ChFinAnn and DuEE-Fin buckets `1`, `2`, `3`,
  `>=4`, showing Role F1, Exact-Record, and Count Accuracy.
- Exact labels to appear:
  - `Role F1`
  - `Exact-Record`
  - `Count Acc.`
  - `1`, `2`, `3`, `>=4`
  - `ChFinAnn`
  - `DuEE-Fin`
- Style guide: code-generated matplotlib/SVG/PDF preferred over image generation to preserve numbers.
- Color palette: use Navy, Teal, Amber, Coral, Slate from the shared palette.
- Layout/aspect ratio: wide single-column or double-column chart depending on page budget.
- Negative prompt:

```text
no fabricated numbers, no decorative chart junk, no 3D bars, no unreadable tick labels, no generated table text
```

- Post-processing plan for exact text overlay: Generate the chart directly from CSV with Python/matplotlib only
  if it is included in the paper; do not use image generation for numerical bars or labels.
