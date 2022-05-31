export default function classNames(
  ...classes: (string | null | undefined | false)[]
): string {
  return classes.filter(Boolean).join(" ");
}
