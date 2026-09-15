# GPU routing benchmark image

`motis` + `nigiri-benchmark` built for one GPU target, no data inside. At
container start `fetch-data.sh` downloads an already imported timetable
(Transitous' nightly one by default), writes a timetable-only `config.yml`
next to it and generates a fixed `queries.txt`. Made for renting a card on
vast.ai without paying for a build or an import.

## Build and push

```sh
docker login ghcr.io          # PAT with write:packages
docker/bench/build.sh ghcr.io/<user>/motis-bench:cuda --push
```

Build context is the repo root with `deps/` resolved (the root
`.dockerignore` keeps `build/`, `cmake-build-*/`, `.git` and `docker/` out;
the scripts come in through a second context, `--build-context
bench=docker/bench`, so use `build.sh` rather than a bare `docker build` --
this keeps edits here from invalidating the cached compile). GHCR
packages start private: flip the package to public after the first push so
vast.ai hosts can pull it without credentials.

A docker-container builder's default GC keeps ~9 GiB of cache, less than
the compile stage, so every rebuild recompiles (~6 min). A builder with a
bigger budget keeps the compile cached across doc/script edits:

```sh
printf '[worker.oci]\n  gc = true\n  reservedSpace = "40GB"\n  maxUsedSpace = "60GB"\n' > /tmp/buildkitd.toml
docker buildx create --name bench --driver docker-container --buildkitd-config /tmp/buildkitd.toml
BUILDX_BUILDER=bench docker/bench/build.sh ...
```

Build args (`--build-arg NAME=value`):

| arg | default | note |
|---|---|---|
| `BASE_DEVEL` | `nvidia/cuda:13.3.1-devel-ubuntu26.04` | what production runs |
| `BASE_RUNTIME` | `ubuntu:26.04` | plus `libcudart.so.13` copied from the build stage (the nvidia runtime images add ~2.5 GB of math libraries motis never links); needs host driver >= 580 |
| `GCC` | `15` | host compiler (Ubuntu 26.04 stock); CUDA 13.3 accepts GCC 6-15 |
| `CMAKE_GPU_FLAGS` | `-DNIGIRI_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` | sm_120 = RTX 5080 / 5070 Ti; nigiri sets no arch itself, without this the card JITs old PTX |
| `JOBS` | `nproc` | build parallelism |

**Which commit wrote the timetable matters.** `tt.bin` is a cista dump
whose header carries a static type hash of `timetable` plus a checksum;
`cista::read` verifies both and refuses a mismatch with `invalid static
version`. Transitous's nightly (written by their v2.11.x deployment) does
*not* load into a master build -- tested 2026-09-05, the header hashes
differ (`911a2b36…` vs `5fe46219…` written by master) even though no
header reachable from `timetable.h` changed between the two nigiri
commits. So: import with the same image that benchmarks (below), and
never mix binaries and data from different builds.

## Data

### Own import (the way that works)

Mirror the Transitous inputs (~15 GB), strip the config, and run the
import with the image's own `motis` -- the same binary that later loads the
file. `motis batch` additionally wants `queries.txt` (from `motis
generate`) and the `meta/` hashes the import writes, all of which the bake
below carries along. Note `--entrypoint`: the image's default entrypoint
is `fetch-data.sh`, which would start downloading.

```sh
mkdir transitous && cd transitous
wget --limit-rate=30m --mirror -l 2 --no-parent --cut-dirs=1 --no-host-directories \
  --include-directories=gtfs,gtfs/scripts --accept .zip --accept .lua --accept config.yml \
  -e robots=off https://api.transitous.org/gtfs/
# importing an older mirror: set the window to when its feeds were valid
# (--first-day 2026-05-23 --num-days 90); the query generators draw days
# uniformly over the timetable's interval, so a window past the feeds'
# validity yields empty queries
python3 <motis>/docker/bench/strip-config.py config.yml config.bench.yml --gpu-states 4
IMG=ghcr.io/felixguendling/motis-bench:cuda
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/in" -v "$PWD/../motis-data:/data" -w /in \
  --entrypoint /opt/motis/motis $IMG import -c config.bench.yml -d /data
```

The import runs fine without a GPU (2045 datasets, 90 days: ~10 min on 24
cores, 34 GB peak RSS, 14.9 GB `tt.bin`). `motis generate` does not: a
`NIGIRI_CUDA` build of `motis` uploads the timetable to the GPU whenever it
loads data, so `queries.txt` is produced on the instance instead --
`fetch-data.sh` runs `motis generate` when the file is missing, and the
generator is deterministic, so every instance gets the same queries.

Then bake `../motis-data` into the image (next section) -- or serve it
over HTTP and point `DATA_URL` at it.

### Downloading at start

`fetch-data.sh` (also the image entrypoint) is idempotent and resumes:

| env | default | |
|---|---|---|
| `DATA_URL` | `https://api.transitous.org/gtfs/data` | serves `tt.bin` (20.6 GB), `tags.bin`, `config.yml` |
| `DATA_DIR` | `/data` | |
| `GPU_STATES` | motis default (2) | `server.gpu_states`; make it >= the `motis batch` thread count |
| `N_QUERIES` | `1000` | `motis generate` size, `0` to skip |

`strip-config.py` drops everything config validation ties to OSM/street
routing (osm, street_routing, tiles, geocoding, elevators, gbfs, prima), sets
`osr_footpath: false` (so `tt.bin`, not `tt_ext.bin`, is loaded), turns off
shapes/railviz and removes the datasets' `rt` feeds so nothing touches the
network at load time.

Transitous refreshes the file nightly, and serves it at ~6 MB/s (about an
hour). For repeated runs bake the downloaded snapshot into the image
instead -- GHCR pulls onto a datacenter host in a minute or two, and every
instance then benchmarks the identical timetable.

### Baking the data in

Registries cap a *compressed* layer at 10 GB. Transitous's `tt.bin` gzips
about 4:1 (measured 2026-09-05 on 300 MB samples: 3.7-4.4), so the whole
20.6 GB file fits one ~5.5 GB layer: bake with `parts` = 1 and it ships as
`/data/tt.bin` directly, nothing to join, ~30 GB of instance disk. The
default of 4 parts (joined by `fetch-data.sh` on first start, at the price
of a second 20.6 GB copy) is only for files that compress worse.
`bake-data.sh` does it with
[crane](https://github.com/google/go-containerregistry) -- no docker, no
second copy of the file -- so run it right on the instance that already
holds `/data` (its uplink is the fast one):

```sh
curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
  | tar -xz -C /usr/local/bin crane
GHCR_USER=felixguendling GHCR_TOKEN=<PAT with write:packages> \
  /opt/motis/bake-data.sh /data ghcr.io/felixguendling/motis-bench:cuda \
                          ghcr.io/felixguendling/motis-bench:cuda-transitous-$(date +%Y%m%d) 1
```

The slices are gzipped locally (`apt-get install pigz` first: 56 cores
beat one) and handed to crane compressed, so the baking host needs only
one compressed slice of extra disk (~5.5 GB for the whole file). For
another timetable, check first: `head -c 2000000000 tt.bin | gzip -1 | wc
-c` times (size / 2 GB) has to stay under 10 GB for a single layer. Run `/opt/motis/fetch-data.sh` there as
usual; it joins the parts, strips the config and generates the queries
without touching the network.

## vast.ai

Template: image `ghcr.io/<user>/motis-bench:cuda-transitous-<date>` (the
baked one; plain `:cuda` downloads at start instead), launch mode **SSH**
(vast replaces the entrypoint, so nothing runs by itself: run
`/opt/motis/fetch-data.sh` once you are in, or put it in **onstart** -- for
a baked image it only generates `queries.txt`, ~2 min), no ports, disk >=
30 GB (image 0.3 GB + timetable 15 GB, nothing joined). Filter offers for RTX 5080 /
5070 Ti (both sm_120, 16 GB), host CUDA version >= 13.0 (driver >= 580;
older 570 hosts can run the card but not this runtime -- the image's
`NVIDIA_REQUIRE_CUDA=cuda>=13.0` makes such a container refuse to start
rather than fail later), RAM >= 32 GB (`cista::read` freads the whole
20.6 GB file into heap; on top come ~0.2 GB of rt tables and ~130 MB per CPU
raptor thread, so a GPU-only run peaks around 22 GB and a full `--engines
cpu gpu` sweep at 24 threads around 26 GB -- on 32 GB cap the sweep with
`--threads 8`; 48 GB if you don't want to think about it), direct SSH, high download bandwidth. Time in "Loading" (image
pull) is not billed; the download costs per-GB bandwidth at the host's rate.

With the CLI, the same as a query (fields: `cuda_max_good`,
`driver_version`, `cpu_ram` in MB, `disk_space` in GB, `direct_port_count`):

```sh
vastai search offers 'gpu_name in [RTX_5080,RTX_5070_Ti] cuda_max_good>=13.0 cpu_ram>=32768 disk_space>=50 direct_port_count>0' -o dph
```

Locally: `docker run --rm -it --gpus all -v motis-data:/data <image>` fetches
once into the volume.

## Running

```sh
nvidia-smi
# GPU throughput/latency; same --seed = same queries on every card
nigiri-benchmark -p /data/tt.bin --engines gpu -n 1000 --seed 1 --gpu_states 1 2 4
# CPU baseline + per-query cross-check of both engines
nigiri-benchmark -p /data/tt.bin --engines cpu gpu -n 1000 --seed 1
# full motis path; a NIGIRI_CUDA build routes on the GPU automatically
motis batch -d /data -q queries.txt -r responses.txt --n_threads 4
# device footprint of the uploaded timetable while a run is loaded
nvidia-smi --query-gpu=memory.used --format=csv
```

## ROCm variant (AMD, HIP)

`Dockerfile.rocm` builds nigiri with `NIGIRI_HIP` (motis-project/nigiri#402)
on `rocm/dev-ubuntu-24.04:7.2.4` and ships a plain-Ubuntu runtime with
`hip-runtime-amd` from AMD's apt repo (~1 GB, not the 7 GB "complete"
image). Host code compiles with g++-14, not the g++-15 production uses: AMD ships
ROCm only for Ubuntu 22.04/24.04 (apt dists `jammy`, `noble`), and neither
carries gcc-15. The GPU kernels are compiled by ROCm's clang either way, so
the host compiler only touches the CPU side. ROCm's clang 20 cannot be the
host compiler itself (it rejects a `fmt::join` over a `basic_string` of
strong ids in `loader/gtfs/seated.cc`), and g++ cannot *link* the
executables because CMake propagates `hip::device`'s `--hip-link` through
the static `nigiri-gpu` library -- so `hip-link.cmake` (injected via
`CMAKE_PROJECT_INCLUDE`) sets `LINKER_LANGUAGE HIP` on `motis` and
`nigiri-benchmark`: g++ compiles, the HIP toolchain links. RX 9070 XT is `gfx1201` (`HIP_ARCHS` build arg), supported from
ROCm 7.0. The PR branch must carry the same `timetable` layout as the CUDA
image's nigiri, or the shared data layer won't load -- build it from
master with the PR applied on top:

```sh
# throwaway worktree with the PR applied (no commit needed)
git worktree add --detach /tmp/motis-rocm master
cp -a deps docker .dockerignore /tmp/motis-rocm/
git -C /tmp/motis-rocm/deps/nigiri fetch origin pull/402/head:pr-402
git -C /tmp/motis-rocm/deps/nigiri checkout --detach origin/master
git -C /tmp/motis-rocm/deps/nigiri merge --no-commit --no-ff pr-402
# the PR also pins cista to felixguendling/cista 98ec4dd (cista/gpu_compat.h)
git -C /tmp/motis-rocm/deps/cista fetch git@github.com:felixguendling/cista.git 98ec4dd798a0ea05526585623d9f0e3968930bed
git -C /tmp/motis-rocm/deps/cista checkout --detach 98ec4dd798a0ea05526585623d9f0e3968930bed
CONTEXT=/tmp/motis-rocm DOCKERFILE=Dockerfile.rocm docker/bench/build.sh ghcr.io/<user>/motis-bench:rocm
```

Known gap in the PR as of `980893ef` (2026-09-05): the rt-transport scan in
`raptor_impl.cuh` still calls `__shfl_up_sync` / `__shfl_sync` /
`__any_sync` directly with the 32-bit `kAllLanes`, and ROCm 7.2's `_sync`
intrinsics static-assert a 64-bit mask. Routing those nine calls through
the PR's own `warp_shfl_up` / `warp_shfl` / `warp_any` (identical codegen
on CUDA) fixes the build; it belongs in the rebase.

Data: re-parent the already pushed timetable layer instead of baking again
(nothing is re-uploaded):

```sh
crane rebase ghcr.io/<user>/motis-bench:cuda-transitous-<date> \
  --old_base ghcr.io/<user>/motis-bench:cuda \
  --new_base ghcr.io/<user>/motis-bench:rocm \
  --tag ghcr.io/<user>/motis-bench:rocm-transitous-<date>
```

The host needs only the amdgpu kernel driver (RDNA4 support: Linux 6.12+
or amdgpu-dkms); everything else is in the image. One command:

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -e N_QUERIES=0 ghcr.io/<user>/motis-bench:rocm-transitous-<date> \
  nigiri-benchmark -p /data/tt.bin --engines gpu --gpu_states 1 2 -n 1000 --seed 1
```

(`rocminfo | grep gfx` inside the container shows whether the card is
visible; `N_QUERIES=0` skips the `motis generate` step the CUDA image uses
for `motis batch`.)
