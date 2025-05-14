// See https://kit.svelte.dev/docs/types#app

import type { Itinerary } from '$lib/api/openapi';

// for information about these interfaces
declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		interface PageState {
			selectedItinerary?: Itinerary;
			selectedStop?: { name: string; stopId: string; time: Date };
			stopArriveBy?: boolean;
			tripId?: string;
			showDepartures?: boolean;
		}
		// interface Platform {}
	}
}

export {};
