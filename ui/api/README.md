# motis-client

Pre-generated JS client for [MOTIS](https://github.com/motis-project/motis) based on the [OpenAPI definition](https://redocly.github.io/redoc/?url=https://raw.githubusercontent.com/motis-project/motis/refs/heads/master/openapi.yaml#tag/routing/operation/plan). See there for parameters, responses and changes between API versions depending on MOTIS versions (correlating to motis-client versions).

For example:

```js
const response = await stoptimes({
	throwOnError: true,
	baseUrl: 'https://api.transitous.org',
	headers: {
		'User-Agent': 'my-user-agent'
	},
	query: {
		stopId: 'de-DELFI_de:06412:7010:1:3',
		n: 10,
		radius: 500
	}
});
console.log(response);
```
