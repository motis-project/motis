## UI Development

### Setup (once)

Install dependencies once before running build/dev commands:

```bash
pnpm install
```

### Build

Build UI (the `-r` is important to also update the OpenAPI client):

```bash
pnpm -r build
```

### Run Dev Server

Run UI development server:

```bash
pnpm dev
```

When developing UI against a local MOTIS backend, open the dev URL with a
`motis` query parameter, for example:

- `http://localhost:5173/?motis=http://localhost:8080`

If Vite selected another port, replace `5173` with the port printed by
`pnpm dev`.

For UI-only changes you can also point to a live backend instead, e.g.
`?motis=https://api.transitous.org`.

Without this parameter, the UI uses the current origin as API base URL, which
leads to 404s for `/api/...` and `/tiles/...` on the Vite server.

### OpenAPI Client

Generate OpenAPI client (when openapi.yaml has been changed, included in `pnpm -r build`):

```bash
pnpm update-api
```

### Publish to npmjs

To publish a new version to npmjs:

```bash
cd src/lib/api
pnpm build
pnpm version patch --no-git-tag-version
pnpm publish --access public
```
