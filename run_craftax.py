import jax
from jax import numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name

rng = jax.random.PRNGKey(0)
rng, _rng = jax.random.split(rng)
rngs = jax.random.split(_rng, 3)

# Create environment
env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
env_params = env.default_params

# Get an initial state and observation
obs, state = env.reset(rngs[0], env_params)
print(f"Initial observation: {obs}, shape : {obs.shape}")
print(f"Initial state: {state}")

# Pick random action
action = env.action_space(env_params).sample(rngs[1])

# Step environment
obs, state, reward, done, info = env.step(rngs[2], state, action, env_params)
print(f"Observation after stepping: {obs}")
print(f"State after stepping: {state}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")