<script lang="ts">
	import maplibregl, { type MapGeoJSONFeature } from 'maplibre-gl';
	import { untrack } from 'svelte';
	import { stops, type Mode } from '@motis-project/motis-client';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { onClickStop } from '$lib/utils';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import Layer from '$lib/map/Layer.svelte';
	import { ensureStopIcons, stopIconId } from './stopIcons';
	import { colors } from '$lib/map/style';

	let {
		map,
		zoom,
		bounds,
		theme
	}: {
		map: maplibregl.Map | undefined;
		zoom: number;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		theme: 'light' | 'dark';
	} = $props();

	// Reuse the base map's label colors so stop names match other map text.
	const labelColors = $derived({
		text: colors[theme].text,
		halo: colors[theme].textHalo
	});

	// Grouping is enabled while zoomed out and disabled from zoom level 13 on.
	const GROUPING_MAX_ZOOM = 16;

	// Icon size factor per mode, mirroring the zoom tiers in which the modes
	// appear: long-distance rail is largest, local modes (tram/bus) smallest.
	const modeIconScale = (mode: Mode | undefined): number => {
		switch (mode) {
			case 'HIGHSPEED_RAIL':
			case 'NIGHT_RAIL':
				return 1.0;
			case 'LONG_DISTANCE':
			case 'COACH':
			case 'REGIONAL_FAST_RAIL':
			case 'REGIONAL_RAIL':
			case 'RAIL':
				return 0.9;
			case 'SUBURBAN':
				return 0.85;
			case 'SUBWAY':
				return 0.8;
			default:
				return 0.8;
		}
	};

	//QUERY
	let query = $derived.by(() => {
		if (!bounds) {
			return null;
		}
		const b = maplibregl.LngLatBounds.convert(bounds);
		const max = lngLatToStr(b.getNorthWest());
		const min = lngLatToStr(b.getSouthEast());
		const grouped = zoom < GROUPING_MAX_ZOOM;
		let modes: Mode[] | undefined = [];
		if (zoom > 7) {
			modes.push('AIRPLANE', 'NIGHT_RAIL', 'HIGHSPEED_RAIL', 'LONG_DISTANCE', 'COACH');
		}
		if (zoom > 11) {
			modes.push('REGIONAL_RAIL', 'FERRY');
		}
		if (zoom > 12) {
			modes.push('SUBWAY');
		}
		if (zoom > 13) {
			modes.push('SUBURBAN');
		}
		if (zoom > 14) {
			modes.push('TRAM');
		}
		if (zoom > 15) {
			modes.push('BUS', 'FUNICULAR', 'AERIAL_LIFT');
		}
		return { min, max, grouped, modes };
	});

	//DATA
	const EMPTY: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] };
	let data = $state<GeoJSON.FeatureCollection>(EMPTY);

	//HANDLERS
	const onLayerClick = (e: { features?: MapGeoJSONFeature[] }) => {
		const props = e.features?.[0]?.properties;
		if (!props?.stopId) {
			return;
		}
		onClickStop(props.name, props.stopId, new Date(Date.now()));
	};

	//UPDATE
	$effect(() => {
		if (!query || !map) {
			data = EMPTY;
			return;
		}
		const currentMap = map;
		untrack(async () => {
			const { data: result } = await stops({ query });
			if (!result) {
				data = EMPTY;
				return;
			}

			const grouped = query.grouped;
			const features: GeoJSON.Feature[] = result.map((s) => ({
				type: 'Feature',
				geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
				properties: {
					stopId: s.stopId,
					name: s.name,
					track: s.track,
					label: !grouped && s.track ? s.track : s.name,
					modes: s.modes ? JSON.stringify(s.modes) : undefined,
					icon: stopIconId(s.modes?.[0]),
					iconSize: modeIconScale(s.modes?.[0]) * (grouped ? 1 : 0.85)
				}
			}));

			await ensureStopIcons(
				currentMap,
				result.flatMap((s) => s.modes ?? [])
			);
			data = { type: 'FeatureCollection', features };
		});
	});
</script>

{#if map}
	<GeoJSON id="stops-view" {data}>
		<Layer
			id="stops-view-layer"
			type="symbol"
			beforeLayerId=""
			filter={['all']}
			layout={{
				'icon-image': ['get', 'icon'],
				'icon-size': ['get', 'iconSize'],
				'symbol-sort-key': ['get', 'iconSize'],
				'icon-allow-overlap': true,
				'icon-ignore-placement': true,
				'text-field': ['get', 'label'],
				'text-font': ['Noto Sans Regular'],
				'text-size': 11,
				'text-offset': [0, 1.1],
				'text-anchor': 'top',
				'text-optional': true
			}}
			paint={{
				'text-color': labelColors.text,
				'text-halo-color': labelColors.halo,
				'text-halo-width': 1.5
			}}
			onclick={onLayerClick}
		/>
	</GeoJSON>
{/if}
