# RSL UI

## Requirements

- [Node.js](https://nodejs.org/) Version 16 or newer
- [pnpm](https://pnpm.io/) Version 7.12 or newer

Install the required dependencies using:

```shell
pnpm install
```

## Production Build

```shell
pnpm run build
```

Output files are written to the `dist` directory.

## Development

### Server

Run a development server with live reload on http://127.0.0.1:5173/rsl/:

```shell
pnpm run dev
```

Run a development server with live reload on http://0.0.0.0:5173/rsl/:

```shell
pnpm run dev-host
```

### Code Formatting

[Prettier](https://prettier.io/) is used to format the source code.
Enable [editor integration](https://prettier.io/docs/en/editors.html)
or format manually by running `pnpm run format`.

### Linter

Run [ESLint](https://eslint.org/) using `pnpm run lint`.
Some problems can be fixed automatically using `pnpm run fix-lint`.

### TypeScript

The development server does not check types. They should be checked
by the editor/IDE or by running `pnpm run ts-check`.
Types are checked in production builds.

### MOTIS API / Protocol

The TypeScript MOTIS API definitions (in `src/api/protocol`) are
generated automatically from the FlatBuffers protocol specification
(see the `protocol` directory in the root project).
To update the TypeScript definitions after modifying the FlatBuffers
files, run `pnpm run update-protocol` (not all types are generated,
see `protocol.config.json`).

## Using the UI

If MOTIS is running on `127.0.0.1:8080`, start the development server
by running `pnpm run dev` and go to http://127.0.0.1:5173/rsl/?motis=8080.

### URL parameters

* `?motis=8082`: Connect to MOTIS on <window.location.hostname>:8082
* `?motis=host`: Connect to MOTIS on host:8080
* `?motis=host:8082`: Connect to MOTIS on host:8082

