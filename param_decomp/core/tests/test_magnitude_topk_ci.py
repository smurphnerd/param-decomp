"""The parameter-free magnitude top-k CI fn: exact-k selection on `|x @ V|`, fed by
component-activation taps that `forward_for_ci` fills, driving the ordinary train step."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import random

from param_decomp.core.checkpoint import make_checkpoint_manager, restore_step, save_state
from param_decomp.core.ci_fn import (
    MagnitudeTopKCIArch,
    MagnitudeTopKCIFn,
    PlacedCIFn,
    build_ci_fn,
    evaluate_ci,
    exact_top_k_mask,
)
from param_decomp.core.components import SiteC, init_component_stacks, project_unit_u_rows
from param_decomp.core.configs import (
    CIMaskedReconLossConfig,
    FaithfulnessLossConfig,
    HiddenActsReconstruction,
    ImportanceMinimalityLossConfig,
    KeepLastNCheckpoints,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.model import (
    PlacedModel,
    component_activation_tap_key,
    forward_for_ci,
    prepare_compute_weights,
)
from param_decomp.core.objective import build_objective
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)
from param_decomp.targets.glu_transformer import glu_site_specs, site_name
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer_taps import site_output_tap_key

LAYER, HEAD, C, K = 2, 1, 32, 5
SITE = site_name(LAYER, "q")


def _setup():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, (SiteC(SITE, C),), query_head=HEAD)
    model = PlacedModel(
        model=tiny_glu_decomposed_lm(cfg, sites, random.PRNGKey(0), query_head=HEAD),
        placement=None,
    )
    arch = MagnitudeTopKCIArch(k=K, has_position_axis=True, output_sites=(SITE,))
    ci_fn = build_ci_fn(arch, model.sites, random.PRNGKey(2))
    assert isinstance(ci_fn, MagnitudeTopKCIFn)
    vu = init_component_stacks(model.sites, random.PRNGKey(1))
    tokens = random.randint(random.PRNGKey(3), (2, 7), 0, cfg.vocab_size)
    return cfg, model, arch, ci_fn, vu, tokens


def test_exact_top_k_mask_keeps_k_with_index_ties():
    values = jnp.array([[0.4, 0.1, 0.4, 0.2], [0.0, 0.0, 0.0, 0.0]])
    mask = exact_top_k_mask(values, 2)
    assert jnp.array_equal(mask.sum(-1), jnp.array([2.0, 2.0]))
    assert jnp.array_equal(mask[0], jnp.array([1.0, 0.0, 1.0, 0.0]))
    assert jnp.array_equal(mask[1], jnp.array([0.0, 0.0, 1.0, 1.0]))


def test_ci_fn_selects_the_k_largest_component_activations():
    _, model, arch, ci_fn, vu, tokens = _setup()
    assert not jax.tree.leaves(eqx.filter(ci_fn, eqx.is_array))
    assert ci_fn.capture_keys == arch.capture_keys == {component_activation_tap_key(SITE)}
    prepared = prepare_compute_weights(model, vu)
    result, taps = forward_for_ci(model, prepared, tokens, ci_fn.capture_keys)
    assert set(taps) == ci_fn.capture_keys and result.captures == {}
    ci = evaluate_ci(PlacedCIFn(fn=ci_fn, placement=None), taps, remat=False)
    lower = np.asarray(ci.lower[SITE], dtype=np.float32)
    assert lower.shape == (2, 7, C)
    assert np.array_equal(lower.sum(-1), np.full((2, 7), K))
    assert np.array_equal(lower, np.asarray(ci.upper[SITE], dtype=np.float32))
    z = np.abs(np.asarray(taps[component_activation_tap_key(SITE)], dtype=np.float32))
    expected_idx = np.argsort(-z, axis=-1, kind="stable")[..., :K]
    for b in range(2):
        for t in range(7):
            assert set(np.flatnonzero(lower[b, t])) == set(expected_idx[b, t])


def test_forward_for_ci_merges_captures_with_component_taps():
    _, model, _, ci_fn, vu, tokens = _setup()
    key = site_output_tap_key(SITE)
    prepared = prepare_compute_weights(model, vu)
    result, taps = forward_for_ci(model, prepared, tokens, ci_fn.capture_keys, frozenset((key,)))
    assert set(result.captures) == {key}
    assert result.captures[key].shape == (2, 7, 8)
    assert set(taps) == ci_fn.capture_keys


def test_train_step_moves_v_projects_u_and_leaves_no_ci_fn_parameters():
    _, model, _, ci_fn, vu, tokens = _setup()
    opt_vu = optax.adam(1e-2)
    opt_ci = optax.adam(1e-3)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    objective = build_objective(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=0.0,
                pnorm=ScheduleConfig(
                    max_val=2.0, points=(Knot(at=0.0, frac=1.0), Knot(at=1.0, frac=0.2))
                ),
            ),
            CIMaskedReconLossConfig(
                coeff=1.0,
                hidden_acts_reconstruction=HiddenActsReconstruction(
                    coeff=1.0, points=(site_output_tap_key(SITE),)
                ),
            ),
        ),
        model.site_names,
    )
    step = make_train_step(
        model_static=model,
        substrate=ForwardSubstrate.of(
            model,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=objective,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=10,
        faithfulness=faithfulness_loss_for(model.model),
        component_projection=project_unit_u_rows,
    )
    # The step donates its state buffers; keep a host copy of V/U for the comparison.
    before = [np.asarray(leaf) for leaf in jax.tree.leaves(eqx.filter(vu, eqx.is_array))]
    new_state, metrics = step(model, state, tokens, random.PRNGKey(9))
    assert jnp.isfinite(metrics["total"]), metrics
    assert float(metrics["grad_norms/summary/ci_fns"]) == 0.0
    assert float(metrics["grad_norms/summary/components"]) > 0.0
    after = jax.tree.leaves(eqx.filter(new_state.decomposition.components, eqx.is_array))
    assert any(not np.allclose(a, np.asarray(b)) for a, b in zip(before, after, strict=True))
    assert new_state.decomposition.ci_fn == ci_fn
    for _name, site in new_state.decomposition.components.sites_items():
        np.testing.assert_allclose(np.linalg.norm(np.asarray(site.U), axis=-1), 1.0, atol=1e-5)


def test_checkpoint_round_trips_a_parameter_free_ci_fn(tmp_path: Path):
    _, _, _, ci_fn, vu, _ = _setup()
    opt_vu = optax.adam(1e-2)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=optax.adam(1e-3).init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            freq_ema=None,
            step=jnp.asarray(3, jnp.int32),
        ),
    )
    mgr = make_checkpoint_manager(tmp_path / "ckpts", KeepLastNCheckpoints(n=1))
    save_state(mgr, 3, state)
    mgr.wait_until_finished()
    restored = restore_step(mgr, state, 3)
    assert restored.decomposition.ci_fn == ci_fn
    for a, b in zip(
        jax.tree.leaves(eqx.filter(state.decomposition.components, eqx.is_array)),
        jax.tree.leaves(eqx.filter(restored.decomposition.components, eqx.is_array)),
        strict=True,
    ):
        assert jnp.array_equal(a, b)


def test_k_above_c_refuses():
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(cfg, (SiteC(SITE, C),), query_head=HEAD)
    with pytest.raises(AssertionError, match="must be in 1..C"):
        build_ci_fn(
            MagnitudeTopKCIArch(k=C + 1, has_position_axis=True, output_sites=(SITE,)),
            sites,
            random.PRNGKey(0),
        )
