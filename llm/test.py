import os

for k, v in os.environ.items():
    if "proxy" in k.lower():
        print(f"{k} = {v}")
