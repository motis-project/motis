import { client } from '@motis-project/motis-client';
import { browser } from '$app/environment';
import type { QuerySerializerOptions } from '@hey-api/client-fetch';

export const prerender = true;

if (browser) {
	const params = new URL(window.location.href).searchParams;
	const defaultProtocol = window.location.protocol;
	const defaultHost = window.location.hostname;
	const defaultPort = '8080';
	const motisParam = params.get('motis');
	let baseUrl = String(window.location.origin + window.location.pathname);
	if (motisParam) {
		if (/^[0-9]+$/.test(motisParam)) {
			baseUrl = defaultProtocol + '//' + defaultHost + ':' + motisParam;
		} else if (!motisParam.includes(':')) {
			baseUrl = defaultProtocol + '//' + motisParam + ':' + defaultPort;
		} else if (!motisParam.startsWith('http:') && !motisParam.startsWith('https:')) {
			baseUrl = defaultProtocol + '//' + motisParam;
		} else {
			baseUrl = motisParam;
		}
	}
	const querySerializer = { array: { explode: false } } as QuerySerializerOptions;
	client.setConfig({ baseUrl, querySerializer }); //`${window.location}`
}
