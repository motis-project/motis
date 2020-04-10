# Use jemalloc for better performance (memory footprint and runtime)

On Ubuntu 18.04:
```bash
sudo apt install libjemalloc-dev libjemalloc1
```

Run path-prepare with:
```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so ./path-prepare ...
```

