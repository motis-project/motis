# Configuration des tuiles de carte

L'UI MOTIS supporte plusieurs sources de tuiles de carte :

## 1. Tuiles MOTIS (par défaut)

Les tuiles sont générées par le serveur MOTIS depuis les données OSM. Aucune configuration supplémentaire n'est nécessaire.

## 2. MapTiler

MapTiler nécessite une clé API pour fonctionner. Pour l'activer, créez un fichier `.env` dans le répertoire `ui/` :
```bash
# Optionnel : surcharger la clé API
VITE_MAPTILER_API_KEY=votre_cle_api_maptiler

# Optionnel : choisir un autre style MapTiler
# Options : openstreetmap, streets-v2, dark-v2, satellite, etc.
VITE_MAPTILER_STYLE=openstreetmap
```

**Note** : MapTiler charge son style complet depuis leur API. Les features MOTIS (indoor, levels, etc.) continueront d'utiliser les tuiles OSM du serveur MOTIS pour la compatibilité.

**Pour désactiver MapTiler** et utiliser les tuiles MOTIS uniquement, définissez :
```bash
VITE_MAPTILER_API_KEY=
```

## 3. Tuiles personnalisées

Pour utiliser une source de tuiles personnalisée :

Créez un fichier `.env` dans le répertoire `ui/` avec :
```bash
VITE_CUSTOM_TILES_URL=https://votre-serveur.com/tiles/{z}/{x}/{y}.mvt
VITE_CUSTOM_GLYPHS_URL=https://votre-serveur.com/glyphs/{fontstack}/{range}.pbf  # Optionnel
VITE_CUSTOM_SPRITE_URL=https://votre-serveur.com/sprite  # Optionnel
```

**Important** : Les tuiles personnalisées doivent utiliser la même structure de `source-layer` que les tuiles MOTIS pour que tous les layers fonctionnent correctement.

## Configuration dans Docker

Pour utiliser MapTiler dans votre conteneur Docker, vous pouvez :

1. Passer la variable d'environnement au conteneur :
   ```bash
   docker run -e VITE_MAPTILER_API_KEY=votre_cle ...
   ```

2. Ou créer un fichier `.env` dans le répertoire `ui/` avant de builder l'image.

## Script bash

Votre script bash n'a pas besoin de modification pour les tuiles - l'UI gère automatiquement la configuration via les variables d'environnement. Le serveur MOTIS continue de servir les tuiles OSM pour les features spéciales (indoor, levels), même si MapTiler est configuré pour le style de base.

