"""Post-update component projections preserve represented weights."""

import jax.numpy as jnp
import numpy as np

from param_decomp.core.components import (
    component_stacks_from_sites,
    project_unit_u_rows,
)


def test_unit_u_rows_projection_preserves_every_site_vu() -> None:
    components = component_stacks_from_sites(
        {
            "a": (
                jnp.arange(12, dtype=jnp.float32).reshape(4, 3) + 1,
                jnp.array([[3.0, 4.0], [0.0, 2.0], [1.0, -2.0]], dtype=jnp.float32),
            ),
            "b": (
                jnp.arange(15, dtype=jnp.float32).reshape(5, 3) - 4,
                jnp.array([[1.0, 2.0], [-2.0, 2.0], [0.5, 0.5]], dtype=jnp.float32),
            ),
        }
    )
    before = {name: np.asarray(site.V @ site.U) for name, site in components.sites_items()}

    projected = project_unit_u_rows(components)

    for name, site in projected.sites_items():
        np.testing.assert_allclose(np.linalg.norm(np.asarray(site.U), axis=-1), 1.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(site.V @ site.U), before[name], rtol=1e-6, atol=1e-6)


def test_unit_u_rows_projection_refuses_a_zero_decoder_row() -> None:
    components = component_stacks_from_sites(
        {
            "a": (
                jnp.ones((3, 2), dtype=jnp.float32),
                jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.float32),
            )
        }
    )

    try:
        project_unit_u_rows(components)
    except Exception as error:
        assert "zero decoder row" in str(error)
    else:
        raise AssertionError("zero decoder row was accepted")
