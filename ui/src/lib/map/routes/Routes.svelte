<script lang="ts">
	import { SvelteMap, SvelteSet } from 'svelte/reactivity';
	import { ArrowLeft, ArrowRight, Download, LoaderCircle } from '@lucide/svelte';
	import { getModeName } from '$lib/getModeName';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import Control from '$lib/map/Control.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import Layer from '$lib/map/Layer.svelte';
	import Popup from '$lib/map/Popup.svelte';
	import polyline from '@mapbox/polyline';
	import { colord } from 'colord';
	import type { Position } from 'geojson';
	import maplibregl from 'maplibre-gl';
	import type { FeatureCollection, LineString, Point } from 'geojson';
	import {
		routeDetails,
		routes,
		type Leg,
		type Place,
		type RouteInfo,
		type RoutePolyline
	} from '@motis-project/motis-client';
	import { getDecorativeColors } from '$lib/map/colors';
	import { t } from '$lib/i18n/translation';
	import { client } from '@motis-project/motis-client';

	let {
		map,
		bounds,
		zoom,
		shapesDebugEnabled = false
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		shapesDebugEnabled?: boolean;
	} = $props();

	const FETCH_PADDING_RATIO = 0.5;

	type RouteDetailsPayload = Awaited<ReturnType<typeof routeDetails>>['data'];
	type RouteResponseData = {
		routes: RouteInfo[];
		polylines: RoutePolyline[];
		stops: Place[];
		zoomFiltered: boolean;
	};
	type RouteEntry = {
		route: RouteInfo;
		arrayIdx: number;
		color: string;
		directionDeg: number | null;
	};
	type PopupSnapshot = {
		lngLat: maplibregl.LngLatLike;
		event: maplibregl.MapMouseEvent;
		features?: maplibregl.MapGeoJSONFeature[];
	};
	type PopupController = {
		open?: (snapshot: PopupSnapshot) => void;
		close?: () => void;
		getSnapshot?: () => PopupSnapshot | null;
		onSnapshotChange?: (snapshot: PopupSnapshot | null) => void;
	};
	type MapViewState = {
		center: [number, number];
		zoom: number;
		bearing: number;
		pitch: number;
	};
	type FocusRouteOptions = {
		closePopup?: () => void;
		popupSnapshot?: PopupSnapshot | null;
		popupRouteIdxs?: number[];
		popupScrollTop?: number;
	};

	let routesData = $state<RouteResponseData | null>(null);
	let loadedBounds = $state<maplibregl.LngLatBounds | null>(null);
	let loadedZoom = $state<number | null>(null);
	let requestToken = 0;
	let hoveredRouteIdx = $state<number | null>(null);
	let downloadingRouteIdx = $state<number | null>(null);
	let focusingRouteIdx = $state<number | null>(null);
	let focusedRouteData = $state<RouteResponseData | null>(null);
	let focusedMapView = $state<MapViewState | null>(null);
	let focusedPopupSnapshot = $state<PopupSnapshot | null>(null);
	let popupRouteIdxs = $state<number[]>([]);
	let focusedPopupRouteIdxs = $state<number[]>([]);
	let focusedPopupScrollTop = $state(0);
	let pendingPopupScrollTop = $state<number | null>(null);
	let popupScrollContainer = $state<HTMLDivElement>();
	let focusRequestToken = 0;
	const popupController: PopupController = {};

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
		routeIndexes: string; // comma-separated because MapLibre stringifies properties
	};

	const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

	const getRouteDisplayProps = (route: RouteInfo | undefined) => {
		if (!route) {
			return { name: '', color: '#000000' };
		}
		const shortNames = Array.from(new Set(route.transitRoutes.map((r) => r.shortName)));
		const name = shortNames.join(', ');
		const apiColor = route.transitRoutes.find((r) => r.color)?.color;
		const color = apiColor ? `#${apiColor}` : getRouteColor(name);
		return { name, color };
	};

	const getTransitRouteName = (route: RouteInfo['transitRoutes'][number]) =>
		route.shortName || route.longName || '—';

	const getRouteIdxsFromFeatures = (features: maplibregl.MapGeoJSONFeature[] = []) => {
		const routeList = routesData?.routes ?? [];
		if (!routeList.length || !features.length) {
			return [] as number[];
		}

		const routeIdxs = new SvelteSet<number>();
		for (const feature of features) {
			const props = feature.properties as RouteFeatureProperties | null;
			const routeIndexes = (props?.routeIndexes ?? '')
				.split(',')
				.map((value) => Number.parseInt(value));
			for (const arrayIdx of routeIndexes) {
				const route = routeList[arrayIdx];
				if (route) {
					routeIdxs.add(route.routeIdx);
				}
			}
		}

		return Array.from(routeIdxs).sort((a, b) => a - b);
	};

	const getPolylineDisplayProps = (routePolyline: RoutePolyline, routeList: RouteInfo[]) => {
		const rdp = getRouteDisplayProps(routeList[routePolyline.routeIndexes[0]]);
		const color = routePolyline.colors.length ? `#${routePolyline.colors[0]}` : rdp.color;
		return { name: rdp.name, color };
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

	const decodePolylines = (data: RouteResponseData | RouteDetailsPayload | null) =>
		data
			? data.polylines.map((segment) =>
					polyline
						.decode(segment.polyline.points, segment.polyline.precision)
						.map(([lat, lng]) => [lng, lat] as Position)
				)
			: [];

	const getMapViewState = (): MapViewState | null => {
		if (!map) {
			return null;
		}
		const center = map.getCenter();
		return {
			center: [center.lng, center.lat],
			zoom: map.getZoom(),
			bearing: map.getBearing(),
			pitch: map.getPitch()
		};
	};

	const fitRouteData = (data: RouteResponseData) => {
		if (!map) {
			return;
		}
		const routeBounds = new maplibregl.LngLatBounds();
		let hasCoordinates = false;
		for (const polylineCoords of decodePolylines(data)) {
			for (const coordinate of polylineCoords) {
				routeBounds.extend(coordinate as [number, number]);
				hasCoordinates = true;
			}
		}
		if (!hasCoordinates) {
			for (const stop of data.stops) {
				routeBounds.extend([stop.lon, stop.lat]);
				hasCoordinates = true;
			}
		}
		if (hasCoordinates) {
			map.fitBounds(routeBounds, {
				padding: 80,
				duration: 0,
				maxZoom: 16
			});
		}
	};

	const applyFocusedRoute = (data: RouteResponseData, options: FocusRouteOptions = {}) => {
		focusedMapView = getMapViewState();
		focusedPopupSnapshot = options.popupSnapshot ?? null;
		focusedPopupRouteIdxs = options.popupRouteIdxs ?? [];
		focusedPopupScrollTop = options.popupScrollTop ?? 0;
		hoveredRouteIdx = null;
		focusedRouteData = data;
		options.closePopup?.();
		fitRouteData(data);
	};

	const clearFocusedRoute = () => {
		if (map && focusedMapView) {
			map.jumpTo(focusedMapView);
		}
		focusedRouteData = null;
		hoveredRouteIdx = null;
		focusedMapView = null;
		if (focusedPopupSnapshot) {
			popupRouteIdxs = [...focusedPopupRouteIdxs];
			popupController.open?.(focusedPopupSnapshot);
			pendingPopupScrollTop = focusedPopupScrollTop;
		}
		focusedPopupSnapshot = null;
		focusedPopupRouteIdxs = [];
	};

	const loadFocusedRoute = async (routeIdx: number) => {
		const { data, error, response } = await routeDetails({ query: { routeIdx } });
		if (error || !data) {
			throw new Error(
				(typeof error === 'string' && error) ||
					(response?.status ? `HTTP ${response.status}` : 'route details request failed')
			);
		}
		return data;
	};

	const focusRoute = async (entry: RouteEntry, closePopup: () => void) => {
		const token = ++focusRequestToken;
		focusingRouteIdx = entry.route.routeIdx;
		try {
			const data = await loadFocusedRoute(entry.route.routeIdx);
			if (token !== focusRequestToken) {
				return;
			}
			applyFocusedRoute(data, {
				closePopup,
				popupSnapshot: popupController.getSnapshot?.() ?? null,
				popupRouteIdxs: [...popupRouteIdxs],
				popupScrollTop: popupScrollContainer?.scrollTop ?? 0
			});
		} catch (error) {
			console.error('[Routes] failed to load full route details', error);
		} finally {
			if (token === focusRequestToken) {
				focusingRouteIdx = null;
			}
		}
	};

	const closeFocusedRoute = () => {
		clearFocusedRoute();
	};

	const handleRouteRowKeydown = (
		event: KeyboardEvent,
		entry: RouteEntry,
		closePopup: () => void
	) => {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			void focusRoute(entry, closePopup);
		}
	};

	const formatRouteNames = (route: RouteInfo) =>
		route.transitRoutes.map((tr) => tr.shortName || tr.longName).join(', ') || '—';

	const getStopKey = (stop: Place) => stop.stopId || `${stop.lat},${stop.lon}`;

	const getProjectedSegmentDistance = (
		clickPoint: maplibregl.Point,
		start: maplibregl.Point,
		end: maplibregl.Point
	) => {
		const deltaX = end.x - start.x;
		const deltaY = end.y - start.y;
		const segmentLengthSq = deltaX * deltaX + deltaY * deltaY;
		if (segmentLengthSq === 0) {
			const clickDeltaX = clickPoint.x - start.x;
			const clickDeltaY = clickPoint.y - start.y;
			return {
				distanceSq: clickDeltaX * clickDeltaX + clickDeltaY * clickDeltaY,
				angleDeg: null
			};
		}

		const projection =
			((clickPoint.x - start.x) * deltaX + (clickPoint.y - start.y) * deltaY) / segmentLengthSq;
		const segmentFactor = clamp(projection, 0, 1);
		const closestX = start.x + segmentFactor * deltaX;
		const closestY = start.y + segmentFactor * deltaY;
		const distanceX = clickPoint.x - closestX;
		const distanceY = clickPoint.y - closestY;

		return {
			distanceSq: distanceX * distanceX + distanceY * distanceY,
			angleDeg: (Math.atan2(deltaY, deltaX) * 180) / Math.PI
		};
	};

	const featureContainsRoute = (feature: maplibregl.MapGeoJSONFeature, arrayIdx: number) => {
		const props = feature.properties as RouteFeatureProperties | null;
		return (props?.routeIndexes ?? '').split(',').some((idx) => Number.parseInt(idx) === arrayIdx);
	};

	const getFeatureDirectionAtPoint = (
		features: maplibregl.MapGeoJSONFeature[],
		arrayIdx: number,
		clickPoint: maplibregl.PointLike
	): number | null => {
		let bestDistanceSq = Number.POSITIVE_INFINITY;
		let bestAngleDeg: number | null = null;

		for (const feature of features) {
			if (!featureContainsRoute(feature, arrayIdx)) {
				continue;
			}

			const geometry = feature.geometry;
			const candidateLines =
				geometry.type === 'LineString'
					? [geometry.coordinates as Position[]]
					: geometry.type === 'MultiLineString'
						? (geometry.coordinates as Position[][])
						: [];

			for (const line of candidateLines) {
				if (line.length < 2 || !map) {
					continue;
				}
				const point = maplibregl.Point.convert(clickPoint);
				for (let i = 1; i < line.length; i++) {
					const startPoint = map.project(line[i - 1] as [number, number]);
					const endPoint = map.project(line[i] as [number, number]);
					const { distanceSq, angleDeg } = getProjectedSegmentDistance(point, startPoint, endPoint);
					if (distanceSq < bestDistanceSq) {
						bestDistanceSq = distanceSq;
						bestAngleDeg = angleDeg;
					}
				}
			}
		}

		return bestAngleDeg;
	};

	const buildStopFeatures = (
		data: RouteResponseData,
		route: RouteInfo,
		showStopOrder: boolean
	): FeatureCollection<Point> => {
		const stopMap = new SvelteMap<
			string,
			{ lat: number; lon: number; name: string; color: string; stopNumber: string }
		>();
		const { color } = getRouteDisplayProps(route);
		let stopOrder = 1;

		for (const segment of route.segments) {
			const orderedStops = [segment.from, segment.to];
			for (const stopIdx of orderedStops) {
				const stop = data.stops[stopIdx];
				if (!stop) {
					continue;
				}
				const stopId = getStopKey(stop);
				if (!stopMap.has(stopId)) {
					stopMap.set(stopId, {
						lat: stop.lat,
						lon: stop.lon,
						name: stop.name,
						color,
						stopNumber: showStopOrder ? `${stopOrder}` : ''
					});
					++stopOrder;
				}
			}
		}

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
					color: stop.color,
					stopNumber: stop.stopNumber
				}
			}))
		};
	};

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
			`[Routes] received ${data.routes.length} routes, ${data.polylines.length} polylines, ${data.stops.length} stops, zoomFiltered=${data.zoomFiltered}`
		);

		routesData = data;
		loadedBounds = requestBounds;
		loadedZoom = zoom;
	};

	$effect(() => {
		if (map && bounds && !focusedRouteData) {
			const b = maplibregl.LngLatBounds.convert(bounds);
			fetchRoutes(b);
		}
	});

	$effect(() => {
		if (popupScrollContainer && pendingPopupScrollTop !== null) {
			popupScrollContainer.scrollTop = pendingPopupScrollTop;
			pendingPopupScrollTop = null;
		}
	});

	let decodedPolylines = $derived.by(() => decodePolylines(routesData));
	let focusedDecodedPolylines = $derived.by(() => decodePolylines(focusedRouteData));
	let routeIndexesByRouteIdx = $derived.by(
		() => new Map((routesData?.routes ?? []).map((route, arrayIdx) => [route.routeIdx, arrayIdx]))
	);
	let hoveredRoute = $derived.by(() => {
		if (hoveredRouteIdx === null || focusedRouteData) {
			return null;
		}

		const arrayIdx = routeIndexesByRouteIdx.get(hoveredRouteIdx);
		if (arrayIdx === undefined || !routesData) {
			return null;
		}

		return {
			route: routesData.routes[arrayIdx],
			arrayIdx
		};
	});

	let routeFeatures = $derived.by((): FeatureCollection<LineString> => {
		if (!routesData || focusedRouteData) {
			return { type: 'FeatureCollection', features: [] };
		}
		const data = routesData;
		return {
			type: 'FeatureCollection',
			features: data.polylines.map((segment, polylineIdx) => {
				const pdp = getPolylineDisplayProps(segment, data.routes);
				return {
					type: 'Feature',
					geometry: {
						type: 'LineString',
						coordinates: decodedPolylines[polylineIdx] ?? []
					},
					properties: {
						color: pdp.color,
						name: pdp.name,
						routeIndexes: segment.routeIndexes.join(',')
					}
				};
			})
		};
	});

	let hoverRouteFeatures = $derived.by((): FeatureCollection<LineString> => {
		if (!hoveredRoute) {
			return { type: 'FeatureCollection', features: [] };
		}
		const { route, arrayIdx } = hoveredRoute;
		const { name, color } = getRouteDisplayProps(route);
		const { outlineColor, chevronColor } = getDecorativeColors(color);
		return {
			type: 'FeatureCollection',
			features: route.segments.map((segment) => ({
				type: 'Feature',
				geometry: {
					type: 'LineString',
					coordinates: decodedPolylines[segment.polyline] ?? []
				},
				properties: {
					color,
					outlineColor,
					chevronColor,
					name,
					arrayIdx
				}
			}))
		};
	});

	let hoverStopFeatures = $derived.by((): FeatureCollection<Point> => {
		if (!hoveredRoute) {
			return { type: 'FeatureCollection', features: [] };
		}

		return buildStopFeatures(routesData!, hoveredRoute.route, false);
	});

	let focusedRouteFeatures = $derived.by((): FeatureCollection<LineString> => {
		if (!focusedRouteData) {
			return { type: 'FeatureCollection', features: [] };
		}
		const route = focusedRouteData.routes[0];
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
					coordinates: focusedDecodedPolylines[segment.polyline] ?? []
				},
				properties: {
					color,
					outlineColor,
					chevronColor,
					name
				}
			}))
		};
	});

	let focusedStopFeatures = $derived.by((): FeatureCollection<Point> => {
		if (!focusedRouteData) {
			return { type: 'FeatureCollection', features: [] };
		}
		const data = focusedRouteData;
		const route = data.routes[0];
		if (!route) {
			return { type: 'FeatureCollection', features: [] };
		}

		return buildStopFeatures(data, route, true);
	});

	let focusedRoute = $derived.by(() => focusedRouteData?.routes[0] ?? null);

	popupController.onSnapshotChange = (snapshot) => {
		popupRouteIdxs = getRouteIdxsFromFeatures(snapshot?.features);
	};

	const getRouteModeName = (mode: RouteInfo['mode']) => getModeName({ mode } as Leg);

	const getPopupPoint = (lngLat: maplibregl.LngLatLike, fallbackPoint: maplibregl.PointLike) => {
		if (!map) {
			return fallbackPoint;
		}

		return map.project(maplibregl.LngLat.convert(lngLat));
	};

	const getRouteFeaturesAtPoint = (
		lngLat: maplibregl.LngLatLike,
		fallbackPoint: maplibregl.PointLike
	) => {
		const queried = map?.queryRenderedFeatures(getPopupPoint(lngLat, fallbackPoint), {
			layers: ['routes-layer']
		});
		return (queried ?? []) as maplibregl.MapGeoJSONFeature[];
	};

	const getDisplayedRouteIdxs = (
		features: maplibregl.MapGeoJSONFeature[] | undefined,
		routeFeaturesAtPoint: maplibregl.MapGeoJSONFeature[]
	) =>
		popupRouteIdxs.length
			? popupRouteIdxs
			: getRouteIdxsFromFeatures(features?.length ? features : routeFeaturesAtPoint);

	const getRoutesFromRouteIdxs = (
		routeIdxs: number[],
		featuresAtPoint: maplibregl.MapGeoJSONFeature[],
		clickPoint: maplibregl.PointLike
	) => {
		const routeList = routesData?.routes ?? [];
		if (!routeList.length) {
			return [] as RouteEntry[];
		}

		return routeIdxs
			.map((routeIdx) => {
				const arrayIdx = routeIndexesByRouteIdx.get(routeIdx);
				if (arrayIdx === undefined) {
					return null;
				}
				const route = routeList[arrayIdx];
				const featureDirection = getFeatureDirectionAtPoint(featuresAtPoint, arrayIdx, clickPoint);
				return {
					route,
					arrayIdx,
					color: getRouteDisplayProps(route).color,
					directionDeg: featureDirection
				};
			})
			.filter((entry): entry is RouteEntry => entry !== null)
			.sort((a, b) => a.route.routeIdx - b.route.routeIdx);
	};

	function getUrlBase(url: string): string {
		const { origin, pathname } = new URL(url);
		return origin + pathname.slice(0, pathname.lastIndexOf('/') + 1);
	}

	const getDownloadFilename = (contentDisposition: string | null, fallback: string): string => {
		if (!contentDisposition) {
			return fallback;
		}

		const utf8Match = /filename\*=UTF-8''([^;]+)/i.exec(contentDisposition);
		if (utf8Match?.[1]) {
			try {
				return decodeURIComponent(utf8Match[1]).trim();
			} catch {} // eslint-disable-line no-empty
		}

		const plainMatch = /filename="?([^";]+)"?/i.exec(contentDisposition);
		return plainMatch?.[1]?.trim() || fallback;
	};

	const downloadRouteDebug = async (routeIdx: number) => {
		if (downloadingRouteIdx === routeIdx) {
			return;
		}

		downloadingRouteIdx = routeIdx;
		try {
			const apiBaseUrl = getUrlBase(
				client.getConfig().baseUrl
					? client.getConfig().baseUrl + '/'
					: window.location.origin + window.location.pathname
			);
			const response = await fetch(`${apiBaseUrl}api/experimental/shapes-debug/${routeIdx}`);
			if (!response.ok) {
				const body = await response.text();
				throw new Error(body || `HTTP ${response.status}`);
			}

			const blob = await response.blob();
			const filename = getDownloadFilename(
				response.headers.get('content-disposition'),
				`r_${routeIdx}.json.gz`
			);

			const objectUrl = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = objectUrl;
			a.download = filename;
			document.body.appendChild(a);
			a.click();
			a.remove();
			URL.revokeObjectURL(objectUrl);
		} catch (error) {
			console.error('[Routes] failed to download shapes debug data', error);
		} finally {
			downloadingRouteIdx = null;
		}
	};
</script>

{#snippet routesPopup(
	event: maplibregl.MapMouseEvent,
	closePopup: () => void,
	features: maplibregl.MapGeoJSONFeature[] | undefined
)}
	{@const popupPoint = getPopupPoint(event.lngLat, event.point)}
	{@const popupFeatures = features?.length
		? features
		: getRouteFeaturesAtPoint(event.lngLat, event.point)}
	{@const displayedRouteIdxs = getDisplayedRouteIdxs(features, popupFeatures)}
	{@const routesAtPoint = getRoutesFromRouteIdxs(displayedRouteIdxs, popupFeatures, popupPoint)}
	<div
		bind:this={popupScrollContainer}
		class="w-fit max-w-[min(88vw,72rem)] max-h-[min(70vh,32rem)] overflow-y-auto pr-1 text-sm"
		role="dialog"
		tabindex="0"
		onmouseleave={() => {
			hoveredRouteIdx = null;
		}}
	>
		<div class="mb-2 font-semibold">{t.routes(routesAtPoint.length)}</div>
		<table class="w-fit table-auto border-separate border-spacing-y-1 text-sm">
			<thead class="text-xs uppercase text-muted-foreground">
				<tr>
					<th class="w-4 px-1 py-1"></th>
					<th class="w-4 px-1 py-1"></th>
					<th class="px-1 py-1 text-left font-medium">Index</th>
					<th class="px-1 py-1 text-left font-medium">Mode</th>
					<th class="px-1 py-1 text-left font-medium">ID</th>
					<th class="px-1 py-1 text-left font-medium">Name</th>
					<th class="px-1 py-1 text-left font-medium">Stops</th>
					<th class="px-1 py-1 text-left font-medium">Source</th>
					{#if shapesDebugEnabled}
						<th class="px-1 py-1 text-left font-medium">Debug Data</th>
					{/if}
				</tr>
			</thead>
			<tbody>
				{#each routesAtPoint as entry (entry.route.routeIdx)}
					<tr
						class="group cursor-pointer align-top hover:bg-muted/80 focus-within:bg-muted/80"
						role="button"
						tabindex="0"
						aria-label={`Focus route ${entry.route.routeIdx}`}
						onclick={() => {
							void focusRoute(entry, closePopup);
						}}
						onkeydown={(keyboardEvent) => {
							handleRouteRowKeydown(keyboardEvent, entry, closePopup);
						}}
						onmouseenter={() => {
							hoveredRouteIdx = entry.route.routeIdx;
						}}
						onmouseleave={() => {
							hoveredRouteIdx = null;
						}}
					>
						<td class="px-1 py-1 align-top">
							{#if focusingRouteIdx === entry.route.routeIdx}
								<LoaderCircle class="h-3.5 w-3.5 animate-spin text-muted-foreground" />
							{:else}
								<span
									class="inline-block h-2.5 w-2.5 rounded-full"
									style="background: {entry.color}"
								></span>
							{/if}
						</td>
						<td class="px-1 py-1 pr-2 align-top">
							{#if entry.directionDeg !== null}
								<span
									class="inline-flex h-5 w-5 items-center justify-center rounded-full border border-border/70 bg-background/90 text-muted-foreground"
									title="Direction at clicked point"
								>
									<ArrowRight
										class="h-3.5 w-3.5"
										style={`transform: rotate(${entry.directionDeg}deg);`}
									/>
								</span>
							{:else}
								<span class="text-muted-foreground">—</span>
							{/if}
						</td>
						<td class="px-1 py-1 pr-2 align-top whitespace-nowrap">{entry.route.routeIdx}</td>
						<td class="px-1 py-1 pr-2 align-top whitespace-nowrap"
							>{getRouteModeName(entry.route.mode)}</td
						>
						<td class="px-1 py-1 pr-2 align-top">
							{#if entry.route.transitRoutes.length}
								<div class="flex flex-col items-start gap-1">
									{#each entry.route.transitRoutes as tr, i (tr.id + i)}
										<span class="whitespace-nowrap rounded-md bg-muted/60 px-2 py-1 leading-none">
											{tr.id}
										</span>
									{/each}
								</div>
							{:else}
								<span class="text-muted-foreground">—</span>
							{/if}
						</td>
						<td class="px-1 py-1 pr-1 align-top">
							{#if entry.route.transitRoutes.length}
								<div class="flex flex-col items-start gap-1">
									{#each entry.route.transitRoutes as tr, i (tr.id + i)}
										<span class="whitespace-nowrap rounded-md bg-muted/60 px-2 py-1 leading-none">
											{getTransitRouteName(tr)}
										</span>
									{/each}
								</div>
							{:else}
								<span class="text-muted-foreground">—</span>
							{/if}
						</td>
						<td class="px-1 py-1 pr-1 align-top whitespace-nowrap">{entry.route.numStops}</td>
						<td class="px-1 py-1 pr-2 align-top whitespace-nowrap">{entry.route.pathSource}</td>
						{#if shapesDebugEnabled}
							<td class="px-1 py-1 align-top">
								<button
									type="button"
									class="inline-flex h-6 min-w-[7.6rem] items-center justify-center gap-1 rounded border border-border/80 bg-background/85 px-2 text-[11px] leading-none text-foreground transition-colors hover:bg-background disabled:cursor-wait disabled:opacity-70"
									disabled={downloadingRouteIdx === entry.route.routeIdx}
									aria-busy={downloadingRouteIdx === entry.route.routeIdx}
									onclick={(event) => {
										event.stopPropagation();
										void downloadRouteDebug(entry.route.routeIdx);
									}}
								>
									{#if downloadingRouteIdx === entry.route.routeIdx}
										<LoaderCircle class="h-3.5 w-3.5 animate-spin" />
										<span>Generating...</span>
									{:else}
										<Download class="h-3.5 w-3.5" />
										<span>Debug</span>
									{/if}
								</button>
							</td>
						{/if}
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
		<Popup trigger="click" children={routesPopup} controller={popupController} />
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
			'text-max-angle': 30,
			'symbol-sort-key': -100,
			'text-allow-overlap': false
		}}
		paint={{
			'text-color': '#000000',
			'text-halo-color': '#ffffff',
			'text-halo-width': 2
		}}
	/>
</GeoJSON>

<GeoJSON id="routes-focused-source" data={focusedRouteFeatures}>
	<Layer
		id="routes-focused-glow"
		type="line"
		filter={['all']}
		beforeLayerId="routes-names"
		paint={{
			'line-color': ['get', 'color'],
			'line-width': 20,
			'line-opacity': 0.22,
			'line-blur': 6
		}}
		layout={{
			'line-cap': 'round',
			'line-join': 'round'
		}}
	/>
	<Layer
		id="routes-focused-outline"
		type="line"
		filter={['all']}
		beforeLayerId="routes-names"
		paint={{
			'line-color': ['get', 'outlineColor'],
			'line-width': 12,
			'line-opacity': 0.92
		}}
		layout={{
			'line-cap': 'round',
			'line-join': 'round'
		}}
	/>
	<Layer
		id="routes-focused"
		type="line"
		filter={['all']}
		beforeLayerId="routes-names"
		paint={{
			'line-color': ['get', 'color'],
			'line-width': 7,
			'line-opacity': 0.98
		}}
		layout={{
			'line-cap': 'round',
			'line-join': 'round'
		}}
	/>
	<Layer
		id="routes-focused-chevrons"
		type="symbol"
		filter={['all']}
		beforeLayerId="routes-names"
		layout={{
			'symbol-placement': 'line',
			'symbol-spacing': 50,
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
			'text-opacity': 0.88,
			'text-halo-color': ['get', 'outlineColor'],
			'text-halo-width': 0.5,
			'text-halo-blur': 0.2
		}}
	/>
</GeoJSON>

<GeoJSON id="routes-focused-stops-source" data={focusedStopFeatures}>
	<Layer
		id="routes-focused-stops"
		type="circle"
		layout={{}}
		filter={['all']}
		paint={{
			'circle-radius': 10,
			'circle-color': 'white',
			'circle-stroke-width': 4,
			'circle-stroke-color': ['get', 'color']
		}}
	/>
	<Layer
		id="routes-focused-stops-order"
		type="symbol"
		filter={['all']}
		layout={{
			'text-field': ['get', 'stopNumber'],
			'text-font': ['Noto Sans Bold'],
			'text-size': 10,
			'text-allow-overlap': true,
			'text-ignore-placement': true
		}}
		paint={{
			'text-color': '#111827'
		}}
	/>
	<Layer
		id="routes-focused-stops-names"
		type="symbol"
		filter={['all']}
		layout={{
			'text-field': ['get', 'name'],
			'text-font': ['Noto Sans Regular'],
			'text-size': 12,
			'text-offset': [0, 1.45],
			'text-anchor': 'top'
		}}
		paint={{
			'text-color': '#000000',
			'text-halo-color': '#ffffff',
			'text-halo-width': 2.2
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

{#if focusedRoute}
	{@const focusedDisplay = getRouteDisplayProps(focusedRoute)}
	<Control position="top-right" class="mt-24 max-w-[min(24rem,calc(100vw-5.5rem))]">
		<div class="rounded-xl border border-border/80 bg-background/95 p-3 shadow-xl backdrop-blur">
			<div class="flex items-center justify-between gap-3">
				<button
					type="button"
					class="inline-flex shrink-0 items-center gap-2 rounded-md border border-border/80 bg-background px-3 py-2 text-xs font-medium transition-colors hover:bg-muted"
					onclick={closeFocusedRoute}
				>
					<ArrowLeft class="h-3.5 w-3.5" />
					<span class="sr-only">Back</span>
				</button>
				<div class="min-w-0 flex-1">
					<div class="flex items-center gap-2">
						<span
							class="inline-block h-2.5 w-2.5 rounded-full"
							style="background: {focusedDisplay.color}"
						></span>
						<div class="text-sm font-semibold leading-tight truncate">
							{formatRouteNames(focusedRoute)}
						</div>
					</div>
				</div>
				{#if shapesDebugEnabled}
					<button
						type="button"
						class="inline-flex shrink-0 items-center gap-2 rounded-md border border-border/80 bg-background px-3 py-2 text-xs font-medium transition-colors hover:bg-muted disabled:cursor-wait disabled:opacity-70"
						disabled={downloadingRouteIdx === focusedRoute.routeIdx}
						aria-busy={downloadingRouteIdx === focusedRoute.routeIdx}
						onclick={() => {
							void downloadRouteDebug(focusedRoute.routeIdx);
						}}
					>
						{#if downloadingRouteIdx === focusedRoute.routeIdx}
							<LoaderCircle class="h-3.5 w-3.5 animate-spin" />
							<span class="sr-only">Generating...</span>
						{:else}
							<Download class="h-3.5 w-3.5" />
							<span class="sr-only">Debug</span>
						{/if}
					</button>
				{/if}
			</div>
			<div class="mt-3 grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-xs leading-5">
				<div class="text-muted-foreground">Index</div>
				<div>{focusedRoute.routeIdx}</div>
				<div class="text-muted-foreground">Mode</div>
				<div>{getRouteModeName(focusedRoute.mode)}</div>
				<div class="text-muted-foreground">IDs</div>
				<div class="break-words">
					{#if focusedRoute.transitRoutes.length}
						{#each focusedRoute.transitRoutes as transitRoute, i (transitRoute.id + i)}
							<div>{transitRoute.id}</div>
						{/each}
					{:else}
						<div>—</div>
					{/if}
				</div>
				<div class="text-muted-foreground">Names</div>
				<div class="break-words">
					{#if focusedRoute.transitRoutes.length}
						{#each focusedRoute.transitRoutes as transitRoute, i (transitRoute.id + i)}
							<div>{transitRoute.shortName || transitRoute.longName}</div>
						{/each}
					{:else}
						<div>—</div>
					{/if}
				</div>
				<div class="text-muted-foreground">Stops</div>
				<div>{focusedRoute.numStops}</div>
				<div class="text-muted-foreground">Source</div>
				<div>{focusedRoute.pathSource}</div>
			</div>
		</div>
	</Control>
{/if}
