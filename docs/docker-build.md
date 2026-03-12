# Docker Compose Build and Run

This build flow lets you compile MOTIS in a container, create a Docker image,
and run the server, all while keeping build artifacts on your host machine.

## Requirements

- Docker and Docker Compose
- An SSH agent with a key that has **no passphrase** (only needed if dependencies are missing)

## Why SSH is needed

MOTIS uses the `pkg` dependency manager which clones dependencies from GitHub via SSH
(`git@github.com:...`). If the `deps/` directory already contains all dependencies from
a previous build, SSH is not required. However, if `pkg` needs to fetch or update
dependencies, it will require SSH access.

## Build Workflow

The build process has two steps:

1. **Compile MOTIS** (`motis-build` service) - Compiles the binary
2. **Create Docker Image** (`motis-image` service) - Packages the binary into `motis:latest`

### Building

```bash
export MOTIS_UID="$(id -u)"
export MOTIS_GID="$(id -g)"
export SSH_AUTH_SOCK

# Build both the binary and the Docker image
docker compose -f docker-compose.build.yml build

# Run the build (compiles and creates image)
docker compose -f docker-compose.build.yml up --build
```

After the build completes:

- The binary is available at: `build/docker-relwithdebinfo/motis`
- The Docker image is available as: `motis:latest`

### Running MOTIS

Once the image is built, you can run MOTIS using the run compose file:

```bash
# Start the MOTIS server
docker compose up -d

# View logs
docker compose logs -f

# Stop the server
docker compose down
```

The server will be available at `http://localhost:8080`. Data is persisted in the `./data` directory on your host.

## Complete Workflow

1. **Build**: `docker compose -f docker-compose.build.yml up --build`
   - Compiles MOTIS binary
   - Creates `motis:latest` Docker image

2. **Run**: `docker compose up -d`
   - Starts MOTIS server on port 8080
   - Data persisted in `./data` directory

3. **Stop**: `docker compose down`

## Notes

- The build disables `buildcache` by default and uses the system timezone
  database to avoid missing `libcrypto.so.1.1` and embedded tzdb issues.
- Runtime image compatibility matters: when building in Ubuntu/glibc, do not run
  the produced binary in Alpine/musl unless you also build a musl-compatible
  binary.
- **SSH is optional**: If the `deps/` directory already contains all dependencies,
  the build will work without SSH. SSH is only needed when `pkg` needs to fetch
  or update dependencies.
- If your SSH agent is not running and dependencies are missing, start an agent
  and add your key with `ssh-add`.
- The `motis-image` service automatically depends on `motis-build`, so the image
  will only be created after successful compilation.
