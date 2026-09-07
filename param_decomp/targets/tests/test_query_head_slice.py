from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.components import SiteC, component_stacks_from_site_arrays
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.nonlinearity import AttentionHeads
from param_decomp.experiments.lm.config import (
    LMDecompositionConfig,
    LMTargetConfig,
    resolve_decomposition,
)
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.targets.glu_transformer import GLUDecomposedModel, glu_site_specs, site_name
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import site_output_tap_key

LAYER = 2
HEAD = 1
C = 32
SITE = site_name(LAYER, "q")


def _head_model():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, (SiteC(SITE, C),), query_head=HEAD)
    return tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0), query_head=HEAD)


def _exact_components(model: GLUDecomposedModel):
    width = model.stacked.attn.head_dim
    start = HEAD * width
    W = model.stacked.attn.wq[LAYER, start : start + width]
    V = jnp.eye(model.embed.shape[1], C)
    U = V.T @ W.T
    return component_stacks_from_site_arrays(model.sites, {SITE: (V, U)})


def test_query_head_site_has_slice_shape_and_exact_faithfulness():
    model = _head_model()
    site = model.sites[0]
    assert (site.d_in, site.d_out, site.C) == (32, 8, C)
    assert isinstance(site.nonlinearity_partition, AttentionHeads)
    assert site.nonlinearity_partition.head_count == 1

    delta = model.weight_deltas(_exact_components(model))["q"]
    np.testing.assert_allclose(delta, 0.0, atol=1e-6)


def test_query_head_mask_changes_only_selected_head():
    model = _head_model()
    components = _exact_components(model)
    prepared = model.prepare_compute_weights(components, placement=None)
    tokens = jnp.arange(10).reshape(2, 5)
    key = site_output_tap_key(SITE)
    shape = (2, 5, C)

    clean = model.clean_forward(tokens, frozenset((key,)), placement=None)
    all_live = model.masked_forward(
        prepared,
        tokens,
        masking=MaterializedMasking(component_masks={SITE: jnp.ones(shape)}),
        placement=None,
        capture_keys=frozenset((key,)),
        remat=False,
    )
    np.testing.assert_allclose(all_live.output, clean.output, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(all_live.captures[key], clean.captures[key], atol=1e-5)

    zeroed = model.masked_forward(
        prepared,
        tokens,
        masking=MaterializedMasking(component_masks={SITE: jnp.zeros(shape)}),
        placement=None,
        capture_keys=frozenset((key,)),
        remat=False,
    )
    clean_q = np.asarray(clean.captures[key]).reshape(2, 5, 4, 8)
    zeroed_q = np.asarray(zeroed.captures[key]).reshape(2, 5, 4, 8)
    np.testing.assert_allclose(zeroed_q[:, :, HEAD], 0.0, atol=1e-6)
    for head in (0, 2, 3):
        np.testing.assert_allclose(zeroed_q[:, :, head], clean_q[:, :, head], atol=1e-6)


def _decomposition(head: int) -> LMDecompositionConfig:
    return LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "glu_transformer_q_head",
                "layer": 14,
                "head": head,
                "C": 512,
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": 1,
                "input_tap": "first_block_resid",
                "d_model": 64,
                "n_blocks": 1,
                "attention": {"kind": "mha", "n_heads": 2},
                "ffn": {"kind": "gelu", "hidden": 128},
            },
        }
    )


def _target() -> LMTargetConfig:
    return LMTargetConfig.model_validate(
        {
            "attention_implementation": "auto",
            "weights_dtype": "bfloat16",
            "spec": {
                "kind": "hf",
                "model_class": "transformers.Qwen3ForCausalLM",
                "model_name": "Qwen/Qwen3-0.6B-Base",
            },
        }
    )


def test_query_head_config_resolves_one_rank_128_site(tmp_path: Path):
    resolved = resolve_decomposition(_target(), _decomposition(3), tmp_path)
    assert isinstance(resolved.target, TargetConfig)
    assert resolved.target.query_head == 3
    assert resolved.target.sites[0].name == "layers.14.self_attn.q_proj"
    assert (resolved.site_specs[0].d_in, resolved.site_specs[0].d_out) == (1024, 128)
    assert resolved.site_specs[0].C == 512


def test_query_head_config_refuses_out_of_bounds_head(tmp_path: Path):
    with pytest.raises(AssertionError, match="query head 16 exceeds n_head 16"):
        resolve_decomposition(_target(), _decomposition(16), tmp_path)
