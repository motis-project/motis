export default function getQueryParameters(): Record<string, string> {
  const params: Record<string, string> = {};
  if (typeof window !== "undefined") {
    window.location.search
      .substring(1)
      .split("&")
      .forEach((p) => {
        const param = p.split("=");
        params[param[0]] = decodeURIComponent(param[1]);
      });
  }
  return params;
}
