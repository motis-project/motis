Generate OpenAPI client:

```bash
cd src/lib/api
npm run generate
```

To publish a new version to npmjs:

```bash
npm run build
npm version patch --no-git-tag-version
npm publish --access public
```
