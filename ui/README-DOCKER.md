# Build Docker pour MOTIS UI

Ce guide explique comment construire et déployer l'image Docker de l'UI MOTIS.

## Prérequis

- Docker installé
- Le fichier `openapi.yaml` doit être présent dans le répertoire parent (`../openapi.yaml`)

## Construction de l'image

### Option 1 : Depuis le répertoire `ui/`

Si vous êtes dans le répertoire `ui/`, vous devez d'abord copier `openapi.yaml` :

```bash
cd ui
cp ../openapi.yaml .
docker build -t motis-ui:latest .
```

### Option 2 : Depuis le répertoire parent `motis/`

```bash
cd /home/epascal/Projects/motis
docker build -f ui/Dockerfile -t motis-ui:latest ui/
```

### Option 3 : Utiliser le script de build

```bash
cd ui
./docker-build.sh
```

Le script construit l'image et l'exporte dans un fichier `.tar` pour faciliter le transfert.

## Export de l'image

Pour exporter l'image dans un fichier `.tar` (utile pour Portainer) :

```bash
docker save motis-ui:latest -o motis-ui.tar
```

## Import de l'image sur Portainer

1. **Via l'interface Portainer** :
   - Allez dans "Images"
   - Cliquez sur "Import image"
   - Uploadez le fichier `motis-ui.tar`

2. **Via la ligne de commande** :
   ```bash
   docker load -i motis-ui.tar
   ```

## Exécution du conteneur

```bash
docker run -d \
  --name motis-ui \
  -p 8080:80 \
  motis-ui:latest
```

L'UI sera accessible sur `http://localhost:8080`

## Configuration pour Portainer

Dans Portainer, lors de la création du conteneur :

- **Image** : `motis-ui:latest` (ou l'image que vous avez importée)
- **Port mapping** : `80:80` (ou `8080:80` pour exposer sur le port 8080)
- **Restart policy** : `Always` (recommandé)

## Variables d'environnement au runtime

### Configuration du backend MOTIS

Vous pouvez configurer l'URL du backend MOTIS via la variable d'environnement `MOTIS_BACKEND_URL` :

```bash
docker run -d \
  --name motis-ui \
  -p 8080:80 \
  -e MOTIS_BACKEND_URL=http://votre-serveur-motis:8090 \
  motis-ui:latest
```

**Par défaut** : `http://build.ciuj-kune.org:8090`

### Proxy CORS

Nginx proxy automatiquement toutes les requêtes `/api/` et `/tiles/` vers le backend MOTIS, ce qui évite les problèmes CORS. Les headers CORS sont automatiquement ajoutés par nginx.

### Configuration MapTiler (runtime)

La clé n’est plus intégrée au bundle : le script d’entrée génère `runtime-config.js` à partir des variables du conteneur. Utilisez `MAPTILER_API_KEY` et `MAPTILER_STYLE` (ou les alias `VITE_MAPTILER_*`).

```bash
docker run -d \
  --name motis-ui \
  -p 8080:80 \
  -e MOTIS_BACKEND_URL=http://votre-serveur-motis:8090 \
  -e MAPTILER_API_KEY=votre_cle \
  -e MAPTILER_STYLE=openstreetmap \
  motis-ui:latest
```

## Dépannage

### L'image ne se construit pas

- Vérifiez que `openapi.yaml` est accessible
- Vérifiez que tous les fichiers nécessaires sont présents
- Consultez les logs de build : `docker build --progress=plain ...`

### L'UI ne se charge pas

- Vérifiez que le conteneur est en cours d'exécution : `docker ps`
- Vérifiez les logs : `docker logs motis-ui`
- Vérifiez que le port est correctement mappé

### Erreurs de build API

- Assurez-vous que `openapi.yaml` est présent et valide
- Vérifiez que les dépendances sont installées correctement

