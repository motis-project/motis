export function extractNumbers(s: string): number[] {
  return [...s.matchAll(/\d+/g)].map((m) => parseInt(m[0]));
}

export function extractNumbersWithRanges(s: string): number[] {
  return [
    ...new Set(
      [...s.matchAll(/(\d+)(-(\d+))?/g)].flatMap((m) => {
        const [, start, , end] = m;
        const startInt = parseInt(start);
        const endInt = parseInt(end ?? start);
        return Array.from(
          { length: endInt - startInt + 1 },
          (_, i) => startInt + i,
        );
      }),
    ),
  ];
}
