// See https://kit.svelte.dev/docs/types#app

import type { Itinerary } from '@motis-project/motis-client';

// for information about these interfaces
declare global {
	interface Window {
		/** Injected at container startup (Docker) or left empty in dev (see VITE_*). */
		__MOTIS_CONFIG__?: {
			maptilerApiKey?: string;
			maptilerStyle?: string;
		};
	}

	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		interface PageState {
			selectedItinerary?: Itinerary;
			selectedStop?: { name: string; stopId: string; time: Date };
			stopArriveBy?: boolean;
			tripId?: string;
			activeTab?: 'connections' | 'departures' | 'isochrones';
			scrollY?: number;
		}
		// interface Platform {}
	}
}

export {};
