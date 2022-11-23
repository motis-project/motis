export interface TypeFilter {
  includes: RegExp[];
  excludes: RegExp[];
}

function typeMatches(patterns: RegExp[], fqtn: string) {
  for (const pattern of patterns) {
    if (pattern.test(fqtn)) {
      return true;
    }
  }
  return false;
}

export function includeType(filter: TypeFilter, fqtn: string): boolean {
  if (filter.includes.length > 0) {
    if (!typeMatches(filter.includes, fqtn)) {
      return false;
    }
  }
  return !typeMatches(filter.excludes, fqtn);
}
