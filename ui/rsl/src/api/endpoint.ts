import getQueryParameters from "../util/queryParameters";

function getApiEndpoint(params: Record<string, string>) {
  const defaultProtocol = window.location.protocol;
  const defaultHost = window.location.hostname;
  const defaultPort = "8080";
  const motisParam = params["motis"] || null;
  let apiEndpoint = String(window.location.origin + window.location.pathname);
  if (motisParam) {
    if (/^[0-9]+$/.test(motisParam)) {
      apiEndpoint = defaultProtocol + "//" + defaultHost + ":" + motisParam;
    } else if (!motisParam.includes(":")) {
      apiEndpoint = defaultProtocol + "//" + motisParam + ":" + defaultPort;
    } else if (
      !motisParam.startsWith("http:") &&
      !motisParam.startsWith("https:")
    ) {
      apiEndpoint = defaultProtocol + "//" + motisParam;
    } else {
      apiEndpoint = motisParam;
    }
  }

  if (!apiEndpoint.endsWith("/")) {
    apiEndpoint += "/";
  }
  return apiEndpoint;
}

const apiEndpoint = getApiEndpoint(getQueryParameters());

export default apiEndpoint;
