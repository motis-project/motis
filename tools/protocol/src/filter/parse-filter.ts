import { TypeFilter } from "./type-filter";

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
            match.length === 1 ? "[^.]*" : ".*"
          );
        } else {
          return part;
        }
      })
      .join("\\.")}$`
  );
}

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

export function parseTypeFilter(config: any): TypeFilter {
  const filter: TypeFilter = { includes: [], excludes: [] };

  if (typeof config === "object") {
    if (typeof config.include === "object") {
      filter.includes = parseTypePatterns(config.include);
    }
    if (typeof config.exclude === "object") {
      filter.excludes = parseTypePatterns(config.exclude);
    }
  }

  return filter;
}
