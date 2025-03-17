import type { Itinerary, PlanResponse } from '$lib/openapi';
import type { Location } from '$lib/Location';

export const updateStartDest = (from: Location, to: Location) => {
	return (r: { data: PlanResponse | undefined; error: unknown }) => {
		if (r.error) throw new Error(String(r.error));

		r.data!.itineraries.forEach((it: Itinerary) => {
			if (it.legs[0].from.name === 'START') {
				it.legs[0].from.name = from.label!;
			}
			if (it.legs[it.legs.length - 1].to.name === 'END') {
				it.legs[it.legs.length - 1].to.name = to.label!;
			}
		});

		return r.data!;
	};
};
