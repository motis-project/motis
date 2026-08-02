import type {
	Itinerary,
	Leg,
	Place,
	PlanResponse,
	Error as ApiError
} from '@motis-project/motis-client';
import type { Location } from '$lib/Location';
import polyline from '@mapbox/polyline';
import type { RequestResult } from '@hey-api/client-fetch';

export const joinInterlinedLegs = (legs: Leg[]): Leg[] => {
	const joinedLegs: Leg[] = [];
	for (let i = 0; i < legs.length; i++) {
		if (legs[i].interlineWithPreviousLeg) {
			const pred = joinedLegs[joinedLegs.length - 1];
			const curr = legs[i];
			pred.intermediateStops!.push({ ...pred.to, switchTo: curr } as Place);
			pred.to = curr.to;
			pred.duration += curr.duration;
			pred.endTime = curr.endTime;
			pred.scheduledEndTime = curr.scheduledEndTime;
			pred.realTime ||= curr.realTime;
			pred.intermediateStops!.push(...curr.intermediateStops!);
			pred.legGeometry = {
				points: polyline.encode(
					[
						...polyline.decode(pred.legGeometry.points, pred.legGeometry.precision),
						...polyline.decode(curr.legGeometry.points, curr.legGeometry.precision)
					],
					pred.legGeometry.precision
				),
				precision: pred.legGeometry.precision,
				length: pred.legGeometry.length + curr.legGeometry.length
			};
		} else {
			joinedLegs.push(legs[i]);
		}
	}
	return joinedLegs;
};

export const updateItinerary = (it: Itinerary, from: Location, to: Location) => {
	const fixupBoundaries = (legs: Leg[]) => {
		if (legs.length === 0) {
			return;
		}
		if (legs[0].from.name === 'START') {
			legs[0].from.name = from.label!;
		}
		if (legs[legs.length - 1].to.name === 'END') {
			legs[legs.length - 1].to.name = to.label!;
		}
	};

	fixupBoundaries(it.legs);
	it.legs = joinInterlinedLegs(it.legs);
	for (const leg of it.legs) {
		if (leg.alternatives) {
			leg.alternatives = leg.alternatives.map((alt) => {
				fixupBoundaries(alt);
				return joinInterlinedLegs(alt);
			});
		}
	}
};

export const preprocessItinerary = (from: Location, to: Location) => {
	return (r: Awaited<RequestResult<PlanResponse, ApiError, false>>): PlanResponse => {
		if (r.error) {
			throw { error: r.error.error, status: r.response?.status };
		}
		r.data.itineraries.forEach((it) => updateItinerary(it, from, to));
		r.data.direct.forEach((it) => updateItinerary(it, from, to));
		return r.data;
	};
};
