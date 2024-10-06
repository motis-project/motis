import { client } from '$lib/openapi';
import { browser } from '$app/environment';

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
	client.setConfig({ baseUrl }); //`${window.location}`
}
