from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from optax._src import numerics
from optax._src import utils


class ScheduleFreeState(NamedTuple):
    x: optax.Params
    z: optax.Params
    t: jax.Array
    base_optimizer_state: optax.OptState


def schedule_free(
    base_optimizer: optax.GradientTransformation,
    beta: float,
    mu_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
    """Wraps an optimizer to make it schedule-free."""
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params: optax.Params) -> ScheduleFreeState:
        return ScheduleFreeState(
            x=jax.tree_map(lambda x: x.astype(mu_dtype), params),
            z=jax.tree_map(lambda x: x.astype(mu_dtype), params),
            t=jnp.zeros([], jnp.int32),
            base_optimizer_state=base_optimizer.init(params),
        )

    def update_fn(
        updates: optax.Updates,
        opt_state: ScheduleFreeState,
        params: optax.Params,
        *args,
        **kwargs
    ) -> Tuple[optax.Updates, ScheduleFreeState]:
        x_curr = opt_state.x
        z_curr = opt_state.z
        t = numerics.safe_int32_increment(opt_state.t)
        base_optimizer_state = opt_state.base_optimizer_state

        z_updates, base_optimizer_state = base_optimizer.update(
            updates, base_optimizer_state, params, *args, **kwargs
        )
        z_next = optax.apply_updates(z_curr, z_updates)
        x_next = jax.tree.map(
            lambda x, z: x * (1 - 1 / t) + z * (1 / t), x_curr, z_next
        )
        y_next = jax.tree.map(lambda x, z: x * beta + z * (1 - beta), x_next, z_next)
        updates = jax.tree.map(lambda y_dash, y: y_dash - y, y_next, params)

        x_next = utils.cast_tree(x_next, mu_dtype)
        z_next = utils.cast_tree(z_next, mu_dtype)
        opt_state = opt_state._replace(
            x=x_next, z=z_next, t=t, base_optimizer_state=base_optimizer_state
        )
        return updates, opt_state

    return optax.GradientTransformation(init_fn, update_fn)