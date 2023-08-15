import { TypeFilter } from "@/filter/type-filter";

const allowedPartCharacters = /^[-_*a-zA-Z0-9]+$/;

function parseTypePattern(pattern: string): RegExp {
  return new RegExp(
    `^${pattern
      .split(".")
      .map((part) => {
        if (!allowedPartCharacters.test(part)) {
          throw new Error(`invalid type filter: ${pattern}`);
        }
        if (part.includes("*")) {
          return part.replace(/\*+/, (match) =>
            match.length === 1 ? "[^.]*" : ".*",
          );
        } else {
          return part;
        }
      })
      .join("\\.")}$`,
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function parseTypePatterns(patterns: any): RegExp[] {
  const res: RegExp[] = [];
  if (typeof patterns === "object") {
    for (const pattern of patterns) {
      if (typeof pattern === "string") {
        res.push(parseTypePattern(pattern));
      }
    }
  }
  return res;
}

export function parseTypeFilter(config: object): TypeFilter {
  const filter: TypeFilter = { includes: [], excludes: [] };

  if (typeof config === "object") {
    if (
      "include" in config &&
      typeof config.include === "object" &&
      config.include !== null
    ) {
      filter.includes = parseTypePatterns(config.include);
    }
    if (
      "exclude" in config &&
      typeof config.exclude === "object" &&
      config.exclude !== null
    ) {
      filter.excludes = parseTypePatterns(config.exclude);
    }
  }

  return filter;
}
