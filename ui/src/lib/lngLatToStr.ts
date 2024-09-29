import maplibregl from 'maplibre-gl';

export function lngLatToStr(pos: maplibregl.LngLatLike) {
	const p = maplibregl.LngLat.convert(pos);
	return `${p.lat},${p.lng}`;
}
