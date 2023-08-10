import path from "path";

import { MarkdownContext } from "@/output/markdown/context";

export function getFilename(ctx: MarkdownContext, namespace: string[]) {
  return path.resolve(ctx.outputDir, ...namespace) + ".md";
}

export function getGFMHeadingAnchor(heading: string) {
  // https://docs.gitlab.com/ee/user/markdown.html#header-ids-and-links
  let id = heading.toLowerCase();
  id = id.replace(/[^-a-z0-9\s]/g, "");
  id = id.replace(/[-\s]+/, "-");
  return id;
}

export function getTypeHref(
  ctx: MarkdownContext,
  fqtn: string[],
  currentNamespace: string | undefined = undefined,
): string | null {
  const refFqtn = fqtn.join(".");
  if (!ctx.types.has(refFqtn)) {
    return null;
  }
  const refNamespace = fqtn.slice(0, -1);
  const refNamespaceStr = refNamespace.join(".");
  if (refNamespaceStr === currentNamespace) {
    const refName = fqtn[fqtn.length - 1];
    return "#" + getGFMHeadingAnchor(refName);
  } else {
    // TODO
    return null;
  }
}

export function getTypeLink(
  ctx: MarkdownContext,
  fqtn: string[],
  currentNamespace: string | undefined = undefined,
) {
  const typeHref = getTypeHref(ctx, fqtn, currentNamespace);
  if (typeHref) {
    const typeName = fqtn[fqtn.length - 1];
    return `[${typeName}](${typeHref})`;
  } else {
    return `\`${fqtn.join(".")}\``;
  }
}

export function toSingleLine(input: string): string {
  return input.trim().replace(/[\n\r]+/g, " ");
}
