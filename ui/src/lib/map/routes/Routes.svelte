<script lang="ts">
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
	import { getModeName } from '$lib/getModeName';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import Layer from '$lib/map/Layer.svelte';
	import Popup from '$lib/map/Popup.svelte';
	import polyline from '@mapbox/polyline';
	import { colord } from 'colord';
	import type { Position } from 'geojson';
	import maplibregl from 'maplibre-gl';
	import type { FeatureCollection, LineString, Point } from 'geojson';
	import { routes, type Leg, type RouteInfo } from '@motis-project/motis-client';
	import { getDecorativeColors } from '$lib/map/colors';
	import { t } from '$lib/i18n/translation';

	let {
		map,
		bounds,
		zoom
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
	} = $props();

	const FETCH_PADDING_RATIO = 0.5;

	type RoutesPayload = Awaited<ReturnType<typeof routes>>['data'];

	let routesData = $state<RoutesPayload | null>(null);
	let loadedBounds = $state<maplibregl.LngLatBounds | null>(null);
	let loadedZoom = $state<number | null>(null);
	let requestToken = 0;
	let hoveredArrayIdx = $state<number | null>(null);

	const stringToHash = (str: string) => {
		let hash = 0;
		for (let i = 0; i < str.length; i++) {
			hash = str.charCodeAt(i) + ((hash << 5) - hash);
		}
		return hash;
	};

	const getRouteColor = (name: string) => {
		const hash = stringToHash(name);
		const h = Math.abs(hash % 360);
		return colord({ h, s: 80, l: 60 }).toHex();
	};

	type RouteFeatureProperties = {
		color: string;
		name: string;
		arrayIdx: number;
	};

	const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

	const getRouteDisplayProps = (route: RouteInfo) => {
		const shortNames = Array.from(new Set(route.transitRoutes.map((r) => r.shortName)));
		const name = shortNames.join(', ');
		const apiColor = route.transitRoutes.find((r) => r.color)?.color;
		const color = apiColor ? `#${apiColor}` : getRouteColor(name);
		return { name, color };
	};

	const expandBounds = (value: maplibregl.LngLatBounds) => {
		const sw = value.getSouthWest();
		const ne = value.getNorthEast();
		const padLng = (ne.lng - sw.lng) * FETCH_PADDING_RATIO;
		const padLat = (ne.lat - sw.lat) * FETCH_PADDING_RATIO;
		return new maplibregl.LngLatBounds(
			[clamp(sw.lng - padLng, -180, 180), clamp(sw.lat - padLat, -90, 90)],
			[clamp(ne.lng + padLng, -180, 180), clamp(ne.lat + padLat, -90, 90)]
		);
	};

	const boundsContain = (outer: maplibregl.LngLatBounds | null, inner: maplibregl.LngLatBounds) =>
		!!outer && outer.contains(inner.getSouthWest()) && outer.contains(inner.getNorthEast());

	const fetchRoutes = async (mapBounds: maplibregl.LngLatBounds) => {
		const expandedBounds = expandBounds(mapBounds);
		const isBoundsContained = loadedBounds && boundsContain(loadedBounds, mapBounds);
		const isZoomSufficient =
			!routesData?.zoomFiltered || Math.floor(zoom) <= Math.floor(loadedZoom ?? 0);

		if (isBoundsContained && isZoomSufficient) {
			return;
		}

		const token = ++requestToken;

		const requestWithBounds = async (requestBounds: maplibregl.LngLatBounds) => {
			const max = lngLatToStr(requestBounds.getNorthWest());
			const min = lngLatToStr(requestBounds.getSouthEast());
			console.debug('[Routes] requesting routes', { min, max, zoom });
			return { ...(await routes({ query: { max, min, zoom } })), requestBounds };
		};

		let { data, error, response, requestBounds } = await requestWithBounds(expandedBounds);

		if (token !== requestToken) {
			return;
		}

		if (error && response?.status === 422) {
			console.debug(
				'[Routes] routes request returned 422 for expanded bounds, retrying with map bounds'
			);
			({ data, error, response, requestBounds } = await requestWithBounds(mapBounds));
			if (token !== requestToken) {
				return;
			}
		}

		if (error || !data) {
			console.error('[Routes] routes request failed', error);
			return;
		}
		console.debug(
			`[Routes] received ${data.routes.length} routes, zoomFiltered=${data.zoomFiltered}`
		);

		routesData = data;
		loadedBounds = requestBounds;
		loadedZoom = zoom;
	};

	$effect(() => {
		if (map && bounds) {
			const b = maplibregl.LngLatBounds.convert(bounds);
			fetchRoutes(b);
		}
	});

	let routeFeatures = $derived.by((): FeatureCollection<LineString> => {
		if (!routesData) {
			return { type: 'FeatureCollection', features: [] };
		}
		return {
			type: 'FeatureCollection',
			features: routesData.routes.flatMap((route, arrayIdx) => {
				const { name, color } = getRouteDisplayProps(route);
				return route.segments.map((segment) => ({
					type: 'Feature',
					geometry: {
						type: 'LineString',
						coordinates: polyline
							.decode(segment.polyline.points, segment.polyline.precision)
							.map(([lat, lng]) => [lng, lat] as Position)
					},
					properties: {
						color,
						name,
						arrayIdx
					}
				}));
			})
		};
	});

	let hoverRouteFeatures = $derived.by((): FeatureCollection<LineString> => {
		if (!routesData || hoveredArrayIdx === null) {
			return { type: 'FeatureCollection', features: [] };
		}
		const route = routesData.routes[hoveredArrayIdx];
		if (!route) {
			return { type: 'FeatureCollection', features: [] };
		}
		const { name, color } = getRouteDisplayProps(route);
		const { outlineColor, chevronColor } = getDecorativeColors(color);
		return {
			type: 'FeatureCollection',
			features: route.segments.map((segment) => ({
				type: 'Feature',
				geometry: {
					type: 'LineString',
					coordinates: polyline
						.decode(segment.polyline.points, segment.polyline.precision)
						.map(([lat, lng]) => [lng, lat] as Position)
				},
				properties: {
					color,
					outlineColor,
					chevronColor,
					name,
					arrayIdx: hoveredArrayIdx
				}
			}))
		};
	});

	let hoverStopFeatures = $derived.by((): FeatureCollection<Point> => {
		if (!routesData || hoveredArrayIdx === null) {
			return { type: 'FeatureCollection', features: [] };
		}
		const route = routesData.routes[hoveredArrayIdx];
		if (!route) {
			return { type: 'FeatureCollection', features: [] };
		}

		const stopMap = new SvelteMap<
			string,
			{ lat: number; lon: number; name: string; color: string }
		>();
		const { color } = getRouteDisplayProps(route);

		route.segments.forEach((segment) => {
			[segment.from, segment.to].forEach((stop) => {
				const stopId = stop.stopId || `${stop.lat},${stop.lon}`;
				if (!stopMap.has(stopId)) {
					stopMap.set(stopId, {
						lat: stop.lat,
						lon: stop.lon,
						name: stop.name,
						color
					});
				}
			});
		});

		return {
			type: 'FeatureCollection',
			features: Array.from(stopMap.values()).map((stop) => ({
				type: 'Feature',
				geometry: {
					type: 'Point',
					coordinates: [stop.lon, stop.lat]
				},
				properties: {
					name: stop.name,
					color: stop.color
				}
			}))
		};
	});

	const getRouteModeName = (mode: RouteInfo['mode']) => getModeName({ mode } as Leg);

	const getRouteFeaturesAtPoint = (
		event: maplibregl.MapMouseEvent,
		features?: maplibregl.MapGeoJSONFeature[]
	) => {
		const queried = map?.queryRenderedFeatures(event.point, { layers: ['routes-layer'] });
		return (queried?.length ? queried : (features ?? [])) as maplibregl.MapGeoJSONFeature[];
	};

	const getRoutesFromFeatures = (features: maplibregl.MapGeoJSONFeature[]) => {
		const rd = routesData?.routes;
		if (!rd || !rd.length) {
			return [] as Array<{ route: RouteInfo; arrayIdx: number; color: string }>;
		}
		const indexes = new SvelteSet<number>();
		const colorMap = new SvelteMap<number, string>();
		for (const feature of features) {
			const props = feature.properties as RouteFeatureProperties | null;
			if (props?.arrayIdx !== undefined && props?.color !== undefined) {
				indexes.add(props.arrayIdx);
				colorMap.set(props.arrayIdx, props.color);
			}
		}
		return Array.from(indexes)
			.map((arrayIdx) => {
				const route = rd[arrayIdx];
				return { route, arrayIdx, color: colorMap.get(arrayIdx) };
			})
			.filter(
				(entry): entry is { route: RouteInfo; arrayIdx: number; color: string } => !!entry.route
			)
			.sort((a, b) => a.route.routeIdx - b.route.routeIdx);
	};
</script>

{#snippet routesPopup(
	event: maplibregl.MapMouseEvent,
	_2: () => void,
	features: maplibregl.MapGeoJSONFeature[] | undefined
)}
	{@const routeFeaturesAtPoint = getRouteFeaturesAtPoint(event, features)}
	{@const routesAtPoint = getRoutesFromFeatures(routeFeaturesAtPoint)}
	<div
		class="min-w-[340px] max-w-[750px] max-h-[480px] overflow-y-auto pr-1 text-sm"
		role="dialog"
		tabindex="0"
		onmouseleave={() => {
			hoveredArrayIdx = null;
		}}
	>
		<div class="font-semibold mb-2">{t.routes(routesAtPoint.length)}</div>
		<table class="w-full text-sm border-separate border-spacing-y-1">
			<thead class="text-xs uppercase text-muted-foreground">
				<tr>
					<th class="w-4"></th>
					<th class="text-left font-medium">Index</th>
					<th class="text-left font-medium">Mode</th>
					<th class="text-left font-medium">ID</th>
					<th class="text-left font-medium">Name</th>
					<th class="text-left font-medium">Stops</th>
					<th class="text-left font-medium">Source</th>
				</tr>
			</thead>
			<tbody>
				{#each routesAtPoint as entry (entry.arrayIdx)}
					<tr
						class="align-top hover:bg-muted"
						onmouseenter={() => {
							hoveredArrayIdx = entry.arrayIdx;
						}}
						onmouseleave={() => {
							hoveredArrayIdx = null;
						}}
					>
						<td>
							<span class="inline-block h-2.5 w-2.5 rounded-full" style="background: {entry.color}"
							></span>
						</td>
						<td class="pr-3">{entry.route.routeIdx}</td>
						<td class="pr-3">{getRouteModeName(entry.route.mode)}</td>
						<td class="pr-3 whitespace-nowrap">
							{#if entry.route.transitRoutes.length}
								{#each entry.route.transitRoutes as tr, i (tr.id + i)}
									{tr.id}{#if i < entry.route.transitRoutes.length - 1}<br />{/if}
								{/each}
							{:else}
								<span class="text-muted-foreground">—</span>
							{/if}
						</td>
						<td class="pr-1">
							{#if entry.route.transitRoutes.length}
								{#each entry.route.transitRoutes as tr, i (tr.id + i)}
									{tr.shortName}{#if i < entry.route.transitRoutes.length - 1}<br />{/if}
								{/each}
							{:else}
								<span class="text-muted-foreground">—</span>
							{/if}
						</td>
						<td class="pr-1">{entry.route.numStops}</td>
						<td class="pr-3">{entry.route.pathSource}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/snippet}

<GeoJSON id="routes-source" data={routeFeatures}>
	<Layer
		id="routes-layer"
		type="line"
		filter={['all']}
		paint={{
			'line-color': ['get', 'color'],
			'line-width': 3,
			'line-opacity': 0.7
		}}
		layout={{
			'line-cap': 'round',
			'line-join': 'round'
		}}
	>
		<Popup trigger="click" children={routesPopup} />
	</Layer>
	<Layer
		id="routes-names"
		type="symbol"
		filter={['all']}
		layout={{
			'symbol-placement': 'line',
			'text-field': ['get', 'name'],
			'text-font': ['Noto Sans Regular'],
			'text-size': 12,
			'text-max-angle': 30
		}}
		paint={{
			'text-color': '#000000',
			'text-halo-color': '#ffffff',
			'text-halo-width': 2
		}}
	/>
</GeoJSON>

<GeoJSON id="routes-hover-source" data={hoverRouteFeatures}>
	<Layer
		id="routes-hover-outline"
		type="line"
		filter={['all']}
		beforeLayerId="routes-names"
		paint={{
			'line-color': '#ffffff',
			'line-width': 20,
			'line-opacity': 0.9
		}}
		layout={{
			'line-cap': 'round',
			'line-join': 'round'
		}}
	/>
	<Layer
		id="routes-hover"
		type="line"
		filter={['all']}
		beforeLayerId="routes-names"
		paint={{
			'line-color': ['get', 'color'],
			'line-width': 14,
			'line-opacity': 0.95
		}}
		layout={{
			'line-cap': 'round',
			'line-join': 'round'
		}}
	/>
	<Layer
		id="routes-hover-chevrons"
		type="symbol"
		filter={['all']}
		beforeLayerId="routes-names"
		layout={{
			'symbol-placement': 'line',
			'symbol-spacing': 40,
			'text-field': '›',
			'text-size': 24,
			'text-font': ['Noto Sans Bold'],
			'text-keep-upright': false,
			'text-allow-overlap': true,
			'text-rotation-alignment': 'map',
			'text-offset': [0, -0.1]
		}}
		paint={{
			'text-color': ['get', 'chevronColor'],
			'text-opacity': 0.85,
			'text-halo-color': ['get', 'outlineColor'],
			'text-halo-width': 0.5,
			'text-halo-blur': 0.2
		}}
	/>
</GeoJSON>

<GeoJSON id="routes-hover-stops-source" data={hoverStopFeatures}>
	<Layer
		id="routes-hover-stops"
		type="circle"
		layout={{}}
		filter={['all']}
		paint={{
			'circle-radius': 5,
			'circle-color': 'white',
			'circle-stroke-width': 4,
			'circle-stroke-color': ['get', 'color']
		}}
	/>
	<Layer
		id="routes-hover-stops-names"
		type="symbol"
		filter={['all']}
		layout={{
			'text-field': ['get', 'name'],
			'text-font': ['Noto Sans Regular'],
			'text-size': 12,
			'text-offset': [0, 1],
			'text-anchor': 'top'
		}}
		paint={{
			'text-color': '#000000',
			'text-halo-color': '#ffffff',
			'text-halo-width': 2
		}}
	/>
</GeoJSON>
