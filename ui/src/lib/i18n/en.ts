import type { Translations } from './translation';

const translations: Translations = {
	from: 'From',
	to: 'To',
	arrival: 'Arrival',
	departure: 'Departure',
	duration: 'Duration',
	arrivals: 'Arrivals',
	later: 'later',
	earlier: 'earlier',
	departures: 'Departures',
	switchToArrivals: 'Wechsel zu Ankünften',
	switchToDepartures: 'Wechsel zu Abfahrten',
	track: 'Track',
	arrivalOnTrack: 'Arrival on track',
	tripIntermediateStops: (n: number) => {
		switch (n) {
			case 0:
				return 'No intermediate stops';
			case 1:
				return '1 intermediate stop';
			default:
				return `${n}  intermediate stops`;
		}
	},
	sharingProvider: 'Provider'
};

export default translations;
