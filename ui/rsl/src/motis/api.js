import getQueryParameters from "../util/queryParameters";

function getApiEndpoint(params) {
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

let apiEndpoint = getApiEndpoint(getQueryParameters());
console.log(`apiEndpoint: ${apiEndpoint}`);

function sendRequest(target, contentType, content) {
  return fetch(apiEndpoint, {
    method: "POST",
    body: JSON.stringify({
      destination: { target: target },
      content_type: contentType,
      content: content || {},
    }),
  });
}

export { apiEndpoint, sendRequest };
