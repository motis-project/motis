import { OpenAPI } from "$lib/openapi";
import { browser } from '$app/environment';

export const prerender = true;

if (browser) {
  OpenAPI.BASE = 'http://localhost:8000'; //`${window.location}`
}