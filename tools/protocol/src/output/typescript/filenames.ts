import path from "path";

import { TSContext } from "./context";

export function getFilename(ctx: TSContext, namespace: string[]) {
  return path.resolve(ctx.outputDir, ...namespace) + ".ts";
}
