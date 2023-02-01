# Protocol Tool for API Documentation + Type Definitions

## Requirements

- [Node.js](https://nodejs.org/) Version 16 or newer
- [pnpm](https://pnpm.io/) Version 7.15 or newer

Install the required dependencies using:

```shell
pnpm install
```

## Usage

To generate the output files run:

```shell
pnpm start
```

See `protocol.config.yaml` for information about the input and output files.
The types are read from the FlatBuffers schemas. Documentation for the types and paths
is read from the YAML files in `docs/api`.

To use another config file run `pnpm start file.yaml`.

To only generate some outputs use `pnpm start --output openapi-3.0 openapi-3.1`
or `pnpm start --skip json-schema` to ignore some outputs.

Running the protocol tool also updates the input `doc` YAML files by adding any new
types and fields with `TODO` placeholders and removing types that no longer exist
in the FlatBuffers schemas.

## Configuration

- `input`: Path to the root FlatBuffers file
- `doc`:
  - `schemas`: Path to the schema/type documentation files
  - `paths`: Path to the paths.yaml file containing the API endpoints
  - `tags`: Path to the tags.yaml file containing information about tags
- `output`: Map for generated outputs

Each output must include a `format` key specifying the output format
(see below for supported formats and their options).

### Type Filters

All output formats support type filters:

- If `include` filters are listed, types must match an entry in the `include` list.
- If `exclude` filters are listed, types matching an entry in the `exclude` list are excluded.

If both `include` and `exclude` filters are specified, only types matching an entry in the `include` list,
but not matching any entries in the `exclude` list are included.

Type filters can include wildcards:

- `*` matches 0 or more characters except `.`
- `**` matches 0 or more characters

## Supported Output Formats

### JSON Schema

Generates [JSON Schema](https://json-schema.org/) (version 2020-12) files for the schema types.

Configuration:

- `format`: `json-schema`
- `dir` (optional): Output directory, one file per type
- `bundle` (optional): Output file, includes all types
- `base-uri`: The JSON Schema Base URI
- `strict-int-types` (bool): Add constraints for min/max values depending on the type width
- `strict-unions` (bool): Force matching `_type` tags for unions
- `number-formats` (bool): Add `format` annotations for numeric types

### OpenAPI

Generates [OpenAPI 3.0](https://spec.openapis.org/oas/v3.0.3) or
[OpenAPI 3.1](https://spec.openapis.org/oas/v3.1.0) files.

Configuration:

- `format`: `openapi`
- `version`: One of the following supported versions:
  - `3.0.3`
  - `3.1.0`
- `file`: Output file
- `base-uri`: The JSON Schema Base URI
- `ids` (boolean): Include `$id` for schema types
- `info`: The info block for the OpenAPI file (must include at least `title` and `version`)
- `externalDocs` (optional): The externalDocs block for the OpenAPI file
- `servers` (optional): The servers block for the OpenAPI file

#### OpenAPI 3.0 Limitations

- Descriptions of fields with custom types are currently missing, because custom types
  are referenced using `$ref` and no other sibling elements are allowed in OpenAPI 3.0.
- Matching union type tags are not enforced by the schema.
- Only one example per type is allowed (the first example is used, all others are ignored).

### TypeScript Type Definitions

Generates TypeScript type definitions for the schema types.

Configuration:

- `format`: `typescript`
- `dir`: Output directory, one file per namespace
- `header` (optional): Header to include in each file
- `import-base` (optional): Absolute base path for imports (e.g. `@/api/protocol`).
  If not set, relative imports are used.
- `prettier` (bool): Format the generated files using prettier.
  The configuration for the output directory is used.

### Markdown

Generates some simple Markdown API documentation.

Configuration:

- `format`: `markdown`
- `dir`: Output directory, one file per namespace

## Documentation Format

The input documentation uses YAML.
Descriptions can include Markdown formatting.

### Schemas

Schema types are split into one file per namespace. Each file is a map with the top level
keys being the type names (without namespace).
Each type has the following properties:

- `description`: Description of the type (ignored if set to `TODO`)
- `examples` (optional): A list of examples.
  Note that OpenAPI 3.0 only supports one example - the first one given will be used.
- `tags` (optional): A list of tags for the type
- `fields`: Only for `table`/`struct` types - a map with field names as keys and
  the following properties for each field (field names are the keys):
  - `description`: Description of the field  (ignored if set to `TODO`).
    Note that OpenAPI 3.0 does not support descriptions or examples for fields with
    custom types (i.e. anything other than numbers, booleans and strings).
  - `examples` (optional): A list of examples for the field.
    Note the restrictions for OpenAPI 3.0 listed above (only one example, only
    for basic types).

### Paths

The paths file lists the available API paths/targets in a map.
The top level keys are the API paths, and each path has the following properties:

- `summary`: Short summary of the operation
- `description` (optional): Longer description of the operation
- `tags` (optional): A list of tags for the operation
- `input` (optional): Full type name for the request. If missing, a `GET` request is used.
- `output`:
  - `type`: Full type name for the response
  - `description`: Description for the response

### Tags

The tags file is a list of tags. Each tag has a `name` and `description`, e.g.:

```yaml
- name: foo
  description: Foo Description
- name: bar
  description: ...
```
