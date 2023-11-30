import { AlertCircle, CheckCircle2, XCircle } from "lucide-react";
import React from "react";

import { Skeleton } from "@/components/ui/skeleton.tsx";

export function StatusIconOk() {
  return <CheckCircle2 className="h-6 w-6 text-green-600" />;
}

export function StatusIconWarning() {
  return <AlertCircle className="h-6 w-6 text-amber-600" />;
}

export function StatusIconError() {
  return <XCircle className="h-6 w-6 text-red-600" />;
}

export function StatusIconSkeleton() {
  return <Skeleton className="h-6 w-6 rounded-full" />;
}
