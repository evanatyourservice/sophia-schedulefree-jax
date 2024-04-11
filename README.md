# sophia-schedulefree-jax

Figured I would gather these two ideas in one place. Code borrowed from 
[levanter](https://github.com/stanford-crfm/levanter/tree/main)'s implementation of Sophia and 
[ameya98's implementation of ScheduleFree](https://gist.github.com/ameya98/7f103501714f4d2fdc0cb793579648d9), 
but with tweaks here and there. Sophia doesn't keep momentum buffer for ScheduleFree. Sophia is slighty improved with 
ability for multiple monte carlo samples for hutchinson's estimator, automatic separate monte carlo samples per device, 
and rademacher instead of normal distribution sampling. ScheduleFree is modified to allow bfloat16 state, but this 
seems to be unstable so default is float32.

Params should be used for training, but opt_state.x should be used for evaluation/inference.

### To use:

vanilla sophia:
```
warmup_fn = optax.linear_schedule(
    init_value=min_learning_rate,
    end_value=learning_rate,
    transition_steps=warmup_steps,
)
decay_fn = optax.linear_schedule(
    init_value=learning_rate,
    end_value=min_learning_rate,
    transition_steps=total_train_steps - warmup_steps,
)
schedule = optax.join_schedules(
    schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
)

tx = sophia(schedule)

updates, opt_state = tx.update(
    grads,
    opt_state,
    params,
    obj_fn=loss_fn,
)
params = optax.apply_updates(params, updates)

eval_params = opt_state.x
```

schedule free sophia:
```
# schedule free uses flat learning rate with warmup

warmup_fn = optax.linear_schedule(
    init_value=min_learning_rate,
    end_value=learning_rate,
    transition_steps=warmup_steps,
)
decay_fn = optax.constant_schedule(learning_rate)
schedule = optax.join_schedules(
    schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
)

tx = sophia_schedule_free(schedule)

updates, opt_state = tx.update(
    grads,
    opt_state,
    params,
    obj_fn=loss_fn,
)
params = optax.apply_updates(params, updates)

eval_params = opt_state.x
```

an example of the weight decay mask:
```
kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: "kernel" in p)


def kernel_mask(params):
    all_false = jax.tree_util.tree_map(lambda _: False, params)
    return kernels.update(lambda _: True, all_false)

    
tx = sophia(schedule, weight_decay=0.01, mask=kernel_mask)
```