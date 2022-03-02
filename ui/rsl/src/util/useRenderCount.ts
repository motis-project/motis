import { useRef } from "react";

export default function useRenderCount(): number {
  const rc = useRef(0);
  return ++rc.current;
}
