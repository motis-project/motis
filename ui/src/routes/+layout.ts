import { client } from '$lib/openapi';
import { browser } from '$app/environment';

export const prerender = true;

const baseUrl = 'http://localhost:7999';

if (browser) {
	client.setConfig({ baseUrl }); //`${window.location}`
}
