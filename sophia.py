from typing import Any, NamedTuple, Optional, Union, Callable

import jax
from jax import numpy as jnp, vmap
from jax.random import PRNGKey
import optax
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.transform import (
    update_moment,
    bias_correction,
    scale_by_learning_rate,
    add_decayed_weights,
)
from optax._src.utils import canonicalize_dtype, cast_tree
from optax._src.base import ScalarOrSchedule, Params

from schedulefree import schedule_free


def sophia(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[Params], Any]]] = None,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
    n_mc_samples: int = 1,
    mu_dtype: Optional[Any] = None,
    pmap_axis_name: Optional[str] = None,
) -> optax.GradientTransformation:
    """Sophia optimizer with hutchinson's estimator for the hessian diagonal.

    Args:
        learning_rate: Scalar or a schedule for the learning rate.
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the hessian diagonal estimates. Remember,
            effective b2 is `1 - (1 - b2) / update_interval`, e.g. default b2 of 0.99
            is effectively adam-style b2 of 0.999 because update_interval is every 10.
        eps: Small constant to avoid division by zero.
        weight_decay: Weight decay coefficient.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip. Note
            that the Adam gradient transformations are applied to all parameters.
        gamma: Normalizing constant for the hessian diagonal.
        clip_threshold: Threshold for clipping updates.
        update_interval: Interval for updating the hessian diagonal.
        n_mc_samples: Number of monte carlo samples for hutchinson's estimator.
        mu_dtype: dtype of the first moment estimates.
        pmap_axis_name: Provide pmap axis name if using pmap to perform separate
            monte carlo samples on each device for hutchinson's estimator for (almost)
            price of one.

    Returns:
        optax.GradientTransformation
    """
    tx = [
        scale_by_sophia_h(
            b1=b1,
            b2=b2,
            eps=eps,
            gamma=gamma,
            clip_threshold=clip_threshold,
            update_interval=update_interval,
            n_mc_samples=n_mc_samples,
            mu_dtype=mu_dtype,
            pmap_axis_name=pmap_axis_name,
        ),
        add_decayed_weights(weight_decay, mask=mask),
        scale_by_learning_rate(learning_rate),
    ]
    return chain(*tx)


def sophia_schedule_free(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[Params], Any]]] = None,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
    n_mc_samples: int = 1,
    schedulefree_state_dtype: Optional[Any] = jnp.float32,
    pmap_axis_name: Optional[str] = None,
) -> optax.GradientTransformation:
    """Sophia optimizer with ScheduleFree.

    Args:
        learning_rate: Scalar or a schedule for the learning rate.
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the hessian diagonal estimates. Remember,
            effective b2 is `1 - (1 - b2) / update_interval`, e.g. default b2 of 0.99
            is effectively adam-style b2 of 0.999 because update_interval is every 10.
        eps: Small constant to avoid division by zero.
        weight_decay: Weight decay coefficient.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip. Note
            that the Adam gradient transformations are applied to all parameters.
        gamma: Normalizing constant for the hessian diagonal.
        clip_threshold: Threshold for clipping updates.
        update_interval: Interval for updating the hessian diagonal.
        n_mc_samples: Number of monte carlo samples for hutchinson's estimator.
        schedulefree_state_dtype: dtype of the ScheduleFree states.
        pmap_axis_name: Provide pmap axis name if using pmap to perform separate
            monte carlo samples on each device for hutchinson's estimator for (almost)
            price of one.

    Returns:
        optax.GradientTransformation
    """
    tx = [
        scale_by_sophia_h(
            b1=0.0,  # disable momentum
            b2=b2,
            eps=eps,
            gamma=gamma,
            clip_threshold=clip_threshold,
            update_interval=update_interval,
            n_mc_samples=n_mc_samples,
            pmap_axis_name=pmap_axis_name,
        ),
        add_decayed_weights(weight_decay, mask=mask),
        scale_by_learning_rate(learning_rate),
    ]
    tx = chain(*tx)
    return schedule_free(tx, beta=b1, mu_dtype=schedulefree_state_dtype)


class ScaleBySophiaState(NamedTuple):
    """State for Sophia and similar."""

    count: jax.Array  # shape=(), dtype=jnp.int32
    mu: Optional[optax.Updates]  # momentum
    nu: optax.Updates  # EMA of hessian diagonal
    key: PRNGKey


def scale_by_sophia_h(
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = 0.01,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
    n_mc_samples: int = 1,
    mu_dtype: Optional[Any] = None,
    pmap_axis_name: Optional[str] = None,
) -> optax.GradientTransformation:
    """Sophia optimizer with hutchinson's estimator for the hessian diagonal.

    Args:
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the hessian diagonal estimates. Remember,
            effective b2 is `1 - (1 - b2) / update_interval`, e.g. default b2 of 0.99
            is effectively adam-style b2 of 0.999 because update_interval is every 10.
        eps: Small constant to avoid division by zero.
        gamma: Normalizing constant for the hessian diagonal.
        clip_threshold: Threshold for clipping updates.
        update_interval: Interval for updating the hessian diagonal.
        n_mc_samples: Number of monte carlo samples for hutchinson's estimator.
        mu_dtype: dtype of the first moment estimates.
        pmap_axis_name: Provide pmap axis name if using pmap to perform separate
            monte carlo samples on each device for hutchinson's estimator for (almost)
            price of one.

    Returns:
        optax.GradientTransformation
    """
    mu_dtype = canonicalize_dtype(mu_dtype)

    def init_fn(params):
        # Don't keep momentum buffer if b1 is 0 to save memory
        if b1 > 0:
            mu = jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
            )
        else:
            mu = None
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        key = jax.random.PRNGKey(0)
        if pmap_axis_name and jax.local_device_count() > 1:
            print(
                "INFO: Using each device as separate monte carlo sample in sophia "
                "optimizer."
            )
            key = jax.random.split(key, jax.local_device_count())
        return ScaleBySophiaState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, key=key)

    def update_fn(updates, state: ScaleBySophiaState, params=None, obj_fn=None):
        count_inc = safe_int32_increment(state.count)

        # If no momentum, replace mu_hat with grads
        if b1 > 0:
            mu = update_moment(updates, state.mu, b1, 1)
            mu_hat = bias_correction(mu, b1, count_inc)
        else:
            mu = None
            mu_hat = updates
        updates = jax.tree_util.tree_map(
            lambda m, h: m / jnp.maximum(gamma * h, eps), mu_hat, state.nu
        )
        if clip_threshold is not None:
            # uncomment to occasionally print percent of updates not clipped
            # authors state this number should stay between 10% and 50%
            # default gamma of 0.01 does seem best
            """percent_not_clipped = jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(
                    lambda u: jnp.sum(jnp.abs(u) <= clip_threshold), updates
                ),
            )
            total_tree_size = sum(x.size for x in jax.tree_util.tree_leaves(updates))
            percent_not_clipped = percent_not_clipped / total_tree_size
            jax.lax.cond(
                count_inc % 2000 == 0,
                lambda: jax.debug.print("{}  {}", count_inc, percent_not_clipped),
                lambda: None,
            )"""

            updates = jax.tree_util.tree_map(
                lambda u: jnp.clip(u, -clip_threshold, clip_threshold), updates
            )

        key, nu = update_hessian(state.key, state.count, state.nu, params, obj_fn)

        if b1 > 0:
            mu = cast_tree(mu, mu_dtype)
        state = ScaleBySophiaState(count=count_inc, mu=mu, nu=nu, key=key)
        return updates, state

    def update_hessian(key, count, nu, params, obj_fn):
        def _do_update(key):
            if pmap_axis_name and jax.local_device_count() > 1:
                # get current replica's key
                idx = jax.lax.axis_index(pmap_axis_name)
                key = jax.lax.dynamic_index_in_dim(key, idx, keepdims=False)

            key, subkey = jax.random.split(key)
            if n_mc_samples > 1:
                mc_keys = jax.random.split(subkey, n_mc_samples)
                hess = vmap(_stochastic_hessian_diagonal, in_axes=(0, None, None))(
                    mc_keys, obj_fn, params
                )
                hess = jax.tree_map(lambda x: jnp.mean(x, axis=0), hess)
            else:
                hess = _stochastic_hessian_diagonal(subkey, obj_fn, params)

            if pmap_axis_name and jax.local_device_count() > 1:
                # mean hessians across devices and gather keys
                hess = jax.lax.pmean(hess, axis_name=pmap_axis_name)
                key = jax.lax.all_gather(key, axis_name=pmap_axis_name)

            # ema of hessian diagonal
            new_nu = update_moment(hess, nu, b2, 1)

            return key, new_nu

        def _dont_update(key):
            return key, nu

        return jax.lax.cond(
            jnp.equal(count % update_interval, 0), _do_update, _dont_update, key
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _tree_gaussian_like(key, tree):
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    # paper uses normal but we use rademacher
    # see https://www.ethanepperly.com/index.php/2024/01/28/dont-use-gaussians-in-stochastic-trace-estimation/
    g = jax.tree_util.tree_map(
        lambda key, x: jax.random.rademacher(key, x.shape, dtype=jnp.float32),
        list(keys),
        leaves,
    )
    g = jax.tree_util.tree_unflatten(structure, g)
    return g


def _stochastic_hessian_diagonal(key, obj_fn, model):
    gaussians = _tree_gaussian_like(key, model)
    product = jax.jvp(jax.grad(obj_fn), (model,), (gaussians,))[1]
    return jax.tree_map(lambda grad, gaussian: grad * gaussian, product, gaussians)
