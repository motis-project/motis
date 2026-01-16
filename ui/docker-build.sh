#!/bin/bash
# Script pour construire et mettre à jour l'image Docker de l'UI MOTIS

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-motis-ui}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
STACK_NAME="${STACK_NAME:-motis-ui}"
SERVICE_NAME="${SERVICE_NAME:-motis-ui_motis-ui}"
MOTIS_BACKEND_URL="${MOTIS_BACKEND_URL:-http://192.168.0.20:8090}"
COMPOSE_REPLICAS="${COMPOSE_REPLICAS:-1}"
COMPOSE_PORT="${COMPOSE_PORT:-80}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

echo "🔨 Construction de l'image Docker: ${IMAGE_NAME}:${IMAGE_TAG}"

# Vérifier que openapi.yaml existe
if [ ! -f "openapi.yaml" ]; then
    if [ -f "../openapi.yaml" ]; then
        echo "📋 Copie de openapi.yaml depuis le répertoire parent..."
        cp ../openapi.yaml ./openapi.yaml
    else
        echo "❌ Erreur: openapi.yaml introuvable"
        echo "   Assurez-vous que le fichier existe dans le répertoire parent ou dans ui/"
        exit 1
    fi
fi

# Charger les variables d'environnement depuis .env si le fichier existe
BUILD_ARGS=""
if [ -f ".env" ]; then
    echo "📋 Chargement des variables d'environnement depuis .env..."
    # Source le fichier .env et extraire les variables VITE_MAPTILER_*
    set -a
    source .env
    set +a
    
    # Construire les arguments --build-arg pour Docker
    if [ -n "${VITE_MAPTILER_API_KEY:-}" ]; then
        BUILD_ARGS="${BUILD_ARGS} --build-arg VITE_MAPTILER_API_KEY=${VITE_MAPTILER_API_KEY}"
    fi
    if [ -n "${VITE_MAPTILER_STYLE:-}" ]; then
        BUILD_ARGS="${BUILD_ARGS} --build-arg VITE_MAPTILER_STYLE=${VITE_MAPTILER_STYLE}"
    fi
fi

# Construire l'image avec les build args
echo "🔨 Build args: ${BUILD_ARGS:-aucun}"
docker build ${BUILD_ARGS} -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "✅ Image construite avec succès"

# Générer le docker-compose.yml compatible Docker Swarm
echo "📝 Génération du fichier docker-compose.yml pour Docker Swarm..."
cat > "${COMPOSE_FILE}" <<EOF
version: '3.8'

services:
  motis-ui:
    image: ${IMAGE_NAME}:${IMAGE_TAG}
    ports:
      - "${COMPOSE_PORT}:80"
    environment:
      - MOTIS_BACKEND_URL=${MOTIS_BACKEND_URL}
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/"]
      interval: 30s
      timeout: 3s
      start_period: 5s
      retries: 3
    deploy:
      replicas: ${COMPOSE_REPLICAS}
      restart_policy:
        condition: any
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
      placement:
        constraints: []
    networks:
      - default

networks:
  default:
    driver: overlay
    attachable: true
EOF

echo "✅ Fichier docker-compose.yml généré: ${COMPOSE_FILE}"

# Vérifier si le service existe déjà
if docker service ls --format '{{.Name}}' | grep -q "^${SERVICE_NAME}$"; then
    echo "🔄 Mise à jour du service Docker Swarm: ${SERVICE_NAME}"
    docker service update --force --image "${IMAGE_NAME}:${IMAGE_TAG}" "${SERVICE_NAME}"
    echo "✅ Service mis à jour avec succès"
else
    echo "📦 Déploiement du nouveau stack Docker Swarm: ${STACK_NAME}"
    docker stack deploy -c "${COMPOSE_FILE}" "${STACK_NAME}"
    echo "✅ Stack déployé avec succès"
fi

