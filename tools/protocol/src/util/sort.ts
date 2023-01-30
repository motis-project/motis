export function sortTypes(types: string[]) {
  types.sort(compareFqtns);
}

export function compareFqtns(as: string, bs: string): number {
  const ap = as.split(".");
  const bp = bs.split(".");
  for (let i = 0; i < Math.min(ap.length, bp.length); ++i) {
    const a = ap[i];
    const b = bp[i];
    if (a === b) {
      continue;
    }
    if (isLowerCase(a) && isUpperCase(b)) {
      return 1;
    } else if (isUpperCase(a) && isLowerCase(b)) {
      return -1;
    } else {
      return b < a ? 1 : -1;
    }
  }
  return ap.length - bp.length;
}

function isLowerCase(c: string) {
  return c >= "a" && c <= "z";
}

function isUpperCase(c: string) {
  return c >= "A" && c <= "Z";
}
