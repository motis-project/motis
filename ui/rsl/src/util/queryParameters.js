export default function getQueryParameters() {
  var params = {};
  window.location.search
    .substr(1)
    .split("&")
    .forEach((p) => {
      var param = p.split("=");
      params[param[0]] = decodeURIComponent(param[1]);
    });
  return params;
}
