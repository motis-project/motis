# 1. Build (compile + create image)
export MOTIS_UID="$(id -u)"
export MOTIS_GID="$(id -g)"
docker compose -f docker-compose.build.yml up --build

# 2. Run
docker compose up -d

# 3. Stop
docker compose down
