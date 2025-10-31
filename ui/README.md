Build UI (the `-r` is important to also update the OpenAPI client):

```bash
pnpm -r build
```

Generate OpenAPI client (when openapi.yaml has been changed, included in `pnpm -r build`):

```bash
pnpm update-api
```

To publish a new version to npmjs:

```bash
cd src/lib/api
pnpm build
pnpm version patch --no-git-tag-version
pnpm publish --access public
```
