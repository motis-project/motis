import { AstMetadata } from "@/fbs/ast";

export function isRequired(metadata: AstMetadata, def = true): boolean {
  if (metadata !== null) {
    for (const attr of metadata) {
      if (attr.id === "optional") {
        return false;
      } else if (attr.id === "required") {
        return true;
      }
    }
  }
  return def;
}
