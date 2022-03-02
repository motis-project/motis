export function isNonNull<T>(x: T | null): x is T {
  return x !== null;
}

export function isDefined<T>(x: T | undefined): x is T {
  return x !== undefined;
}

export function isPresent<T>(x: T | undefined | null): x is T {
  return x != undefined;
}
