export function extractNumbers(s: string): number[] {
  return [...s.matchAll(/\d+/g)].map((m) => parseInt(m[0]));
}
