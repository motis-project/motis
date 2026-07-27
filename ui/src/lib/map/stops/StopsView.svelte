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
	import { LEVEL_MIN_ZOOM } from '$lib/constants';

	let {
		map,
		zoom,
		bounds,
		level,
		theme
	}: {
		map: maplibregl.Map | undefined;
		zoom: number;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		level: number;
		theme: 'light' | 'dark';
	} = $props();

	const labelColors = $derived({
		text: colors[theme].text,
		halo: colors[theme].textHalo
	});

	const GROUPING_MAX_ZOOM = 16;

	// Once the level selector is active (zoom > LEVEL_MIN_ZOOM), only stops on the
	// selected level are shown. Stops without level information count as level 0.
	// Fractional levels (e.g. 0.5) are shown on both adjacent levels.
	const stopLevel: maplibregl.ExpressionSpecification = [
		'coalesce',
		['to-number', ['get', 'level']],
		0
	];
	const levelFilter = $derived<maplibregl.FilterSpecification>(
		zoom > LEVEL_MIN_ZOOM
			? ['any', ['==', ['ceil', stopLevel], level], ['==', ['floor', stopLevel], level]]
			: ['all']
	);

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

	// QUERY
	let query = $derived.by(() => {
		if (!bounds) {
			return null;
		}
		const b = maplibregl.LngLatBounds.convert(bounds);
		const max = lngLatToStr(b.getNorthWest());
		const min = lngLatToStr(b.getSouthEast());
		const grouped = zoom < GROUPING_MAX_ZOOM;
		let modes: Mode[] | undefined = [];
		modes.push('AIRPLANE', 'NIGHT_RAIL', 'HIGHSPEED_RAIL', 'LONG_DISTANCE');
		if (zoom > 9) {
			modes.push('COACH');
		}
		if (zoom > 11) {
			modes.push('REGIONAL_RAIL', 'SUBURBAN', 'FERRY');
		}
		if (zoom > 12) {
			modes.push('SUBWAY');
		}
		if (zoom > 13) {
			modes.push('TRAM');
		}
		if (zoom > 14) {
			modes.push('BUS', 'FUNICULAR', 'AERIAL_LIFT');
		}
		return { min, max, grouped, modes };
	});

	// DATA
	const EMPTY: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] };
	let data = $state<GeoJSON.FeatureCollection>(EMPTY);

	// HANDLERS
	const onLayerClick = (e: { features?: MapGeoJSONFeature[] }) => {
		const props = e.features?.[0]?.properties;
		if (!props?.stopId) {
			return;
		}
		onClickStop(
			props.name,
			props.stopId,
			new Date(Date.now()),
			false,
			false,
			props.exactRadius === true
		);
	};

	// UPDATE
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
			const features: GeoJSON.Feature[] = result.map((s) => {
				// Sort ride-sharing to end of modes (-> use public transport icons)
				const modes: Mode[] | undefined = s.modes?.includes('RIDE_SHARING')
					? [...s.modes.filter((mode) => mode !== 'RIDE_SHARING'), 'RIDE_SHARING']
					: s.modes;
				const mode = modes?.[0];
				return {
					type: 'Feature',
					geometry: { type: 'Point', coordinates: [s.lon, s.lat] },
					properties: {
						stopId: s.stopId,
						name: s.name,
						track: s.track,
						level: s.level ?? 0,
						// Ungrouped stops represent one specific platform / bay / bus stop:
						// clicking them shows the departures of exactly this stop.
						exactRadius: !grouped,
						label: !grouped && s.track ? s.track : s.name,
						modes: modes?.length ? JSON.stringify(modes) : undefined,
						icon: stopIconId(mode),
						iconSize: modeIconScale(mode) * (grouped ? 1 : 0.85) * (zoom < 9 ? 0.6 : 1)
					}
				};
			});

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
			beforeLayerId="stops-anchor"
			filter={levelFilter}
			layout={{
				'icon-image': ['get', 'icon'],
				'icon-size': ['get', 'iconSize'],
				'symbol-sort-key': ['get', 'iconSize'],
				'icon-allow-overlap': true,
				'icon-ignore-placement': true,
				'text-field': ['step', ['zoom'], '', 9, ['get', 'label']],
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
