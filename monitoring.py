import random
from time import sleep
import torch
import cProfile



with cProfile.Profile() as pr:
    sleep(1)
    random_tensor = random.random()

pr.dump_stats("logs/profile_stats.prof")
print("\nProfile stats dumped to profile_stats.prof")
print(
    "You can visualize the profile stats using snakeviz by running 'snakeviz logs/profile_stats.prof'"
)