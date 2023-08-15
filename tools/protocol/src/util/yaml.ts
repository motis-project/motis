import { Document, YAMLMap, isMap, isScalar } from "yaml";

export function removeUnknownKeys(
  oaMap: YAMLMap,
  keep: (key: string) => boolean,
): string[] {
  const unknownKeys: string[] = [];
  for (const pair of oaMap.items) {
    if (!isScalar(pair.key) || typeof pair.key.value !== "string") {
      throw new Error(
        `invalid yaml file: unsupported map key: ${pair.toJSON()}`,
      );
    }
    if (!keep(pair.key.value)) {
      unknownKeys.push(pair.key.value);
    }
  }
  for (const s of unknownKeys) {
    oaMap.delete(s);
  }
  return unknownKeys;
}

export function getOrCreateMap(
  doc: Document,
  parent: YAMLMap | Document,
  path: (string | number)[],
): YAMLMap {
  let map = parent.getIn(path);
  if (map == undefined) {
    map = doc.createNode({});
    parent.setIn(path, map);
  }
  if (!isMap(map)) {
    throw new Error(
      `invalid yaml file: ${
        path.length > 0 ? path.join("/") : "root"
      } is not a map`,
    );
  }
  return map;
}

export function createMap(
  doc: Document,
  parent: YAMLMap | Document,
  path: (string | number)[],
): YAMLMap {
  const map = doc.createNode({});
  parent.setIn(path, map);
  return map;
}
