# Vendored patches

## flashinfer 0.6.6 — `sparse.py`

CONDUIT's attention path relies on flashinfer's variable block-sparse attention
(`VariableBlockSparseAttentionWrapper`) and calls it with an **NHD** KV layout.
In the released 0.6.6, `run()` hard-codes the **HND** `einops.rearrange` and
never consults `self._kv_layout`, so under NHD the tensors come out with the
wrong shape and ordering.

The version in this directory makes both rearranges branch on
`self._kv_layout`:

- on the input side (q / k / v): separate rearrangements for HND and NHD
- on the output side (out): likewise branched on layout. Note that **the HND
  output rearrangement also differs from upstream**: upstream produces
  `-> (num_kv_heads gqa_group_size) qo_len head_dim`, this version produces
  `-> num_kv_heads (qo_len gqa_group_size) head_dim`

Without this patch the reproduction does not yield correct results.

### Files

| File | Description |
| --- | --- |
| `flashinfer-0.6.6/sparse.py` | full replacement file, md5 `b539dec2b4104b5491a53dcbf3c76a5b` |
| `flashinfer-0.6.6/sparse.py.patch` | unified diff against upstream 0.6.6 (2 hunks, 75 lines) |

Upstream project: https://github.com/flashinfer-ai/flashinfer (Apache-2.0)

### Applying it

```bash
pip install flashinfer-python==0.6.6
bash scripts/apply_flashinfer_patch.sh
```

The script is idempotent: it verifies the md5 of the patch file, confirms that
the installed flashinfer is 0.6.6, backs the original up as `sparse.py.orig`,
then overwrites and re-verifies. If the patch is already applied it returns 0
immediately.

With a non-default interpreter:

```bash
PYTHON=/path/to/env/bin/python bash scripts/apply_flashinfer_patch.sh
```

### Or patch by hand

```bash
python -c "import flashinfer, os; print(os.path.dirname(flashinfer.__file__))"
# then, from the parent of the directory printed above:
patch -p1 < patches/flashinfer-0.6.6/sparse.py.patch
```
