import type { Leg } from '@motis-project/motis-client';

const isTransitLeg = (l: Leg) => !!l.displayName;
const ms = (t: string | undefined) => (t === undefined ? undefined : new Date(t).getTime());

// A FLEX ride may start anywhere in its pickup and drop-off windows. The
// street legs attached to it on a side without a transit leg move with it,
// the transit legs stay: a first-mile ride can only leave earlier (it was
// placed to arrive when the trip departs), a last-mile ride only later.
// Returns the admissible start times [lo, hi] (ms) of the ride `legs[i]`;
// lo == hi is a fixed departure. Undefined without windows or when the ride
// lies outside them.
export const flexShiftRange = (legs: Leg[], i: number): { lo: number; hi: number } | undefined => {
	const l = legs[i];
	const start = ms(l.startTime)!;
	const rideDur = ms(l.endTime)! - start;
	const fromStart = ms(l.from.flexStartPickupDropOffWindow);
	const fromEnd = ms(l.from.flexEndPickupDropOffWindow);
	if (fromStart === undefined || fromEnd === undefined) {
		return undefined;
	}
	const toStart = ms(l.to.flexStartPickupDropOffWindow);
	const toEnd = ms(l.to.flexEndPickupDropOffWindow);
	let lo = Math.max(fromStart, toStart === undefined ? -Infinity : toStart - rideDur);
	// Windows are half-open: the last admissible minute is the one before the end.
	let hi = Math.min(fromEnd, toEnd === undefined ? Infinity : toEnd - rideDur) - 60000;
	const prevTransitIdx = legs.slice(0, i).findLastIndex(isTransitLeg);
	const nextTransitIdx = legs.findIndex((x, j) => j > i && isTransitLeg(x));
	if (prevTransitIdx >= 0) {
		const chainStart = ms(legs[prevTransitIdx + 1].startTime)!;
		lo = Math.max(lo, start + ms(legs[prevTransitIdx].endTime)! - chainStart);
	}
	if (nextTransitIdx >= 0) {
		const chainEnd = ms(legs[nextTransitIdx - 1].endTime)!;
		hi = Math.min(hi, start + ms(legs[nextTransitIdx].startTime)! - chainEnd);
	}
	return start < lo || start > hi ? undefined : { lo, hi };
};

// Start times [lo, hi] (ms) of a direct itinerary with a FLEX ride: the whole
// itinerary shifts with the ride.
export const directDepartureRange = (legs: Leg[]): { lo: number; hi: number } | undefined => {
	const i = legs.findIndex((l) => l.mode === 'FLEX');
	if (i < 0 || legs.some(isTransitLeg)) {
		return undefined;
	}
	const range = flexShiftRange(legs, i);
	if (range === undefined) {
		return undefined;
	}
	const offset = ms(legs[i].startTime)! - ms(legs[0].startTime)!;
	return { lo: range.lo - offset, hi: range.hi - offset };
};
