import maplibregl from 'maplibre-gl';

const PRECISION = 7;  // API has ~1cm precision, matching for "query changed" check requires rounding
const round = (v: number) => Number(v.toFixed(PRECISION));

export function lngLatToStr(pos: maplibregl.LngLatLike) {
	const p = maplibregl.LngLat.convert(pos);
	return `${round(p.lat)},${round(p.lng)}`;
}
