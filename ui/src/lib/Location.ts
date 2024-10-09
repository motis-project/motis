import maplibregl from 'maplibre-gl';
import type { Match } from './openapi';

export type Location = {
	label?: string;
	value: {
		match?: Match;
		precision?: number;
	};
};

export function posToLocation(pos: maplibregl.LngLatLike): Location {
	const { lat, lng } = maplibregl.LngLat.convert(pos);
	const label = `${lat},${lng}`;
	return {
		label,
		value: {
			match: {
				lat,
				lon: lng,
				level: 0,
				id: '',
				areas: [],
				type: 'PLACE',
				name: label,
				tokens: [],
				score: 0
			},
			precision: 100
		}
	};
}