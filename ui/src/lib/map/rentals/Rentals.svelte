<script lang="ts">
	import {
		rentals,
		type RentalFormFactor,
		type RentalProvider,
		type RentalStation,
		type RentalVehicle,
		type RentalZone
	} from '$lib/api/openapi';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import Control from '$lib/map/Control.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import Layer from '$lib/map/Layer.svelte';
	import { formFactorAssets, DEFAULT_FORM_FACTOR } from '$lib/map/rentals/assets';
	import { cn } from '$lib/utils';
	import polyline from '@mapbox/polyline';
	import maplibregl from 'maplibre-gl';
	import type { MapLayerMouseEvent } from 'maplibre-gl';
	import { flushSync, mount, onDestroy, unmount } from 'svelte';
	import type {
		Feature,
		FeatureCollection,
		Point,
		Polygon as GeoPolygon,
		MultiPolygon as GeoMultiPolygon,
		Position
	} from 'geojson';
	import StationPopup from '$lib/map/rentals/StationPopup.svelte';
	import VehiclePopup from '$lib/map/rentals/VehiclePopup.svelte';
	import ZonePopup from '$lib/map/rentals/ZonePopup.svelte';
	import {
		zoomScaledIconSize,
		zoomScaledTextOffset,
		zoomScaledTextSizeMedium,
		zoomScaledTextSizeSmall
	} from './style';

	let {
		map,
		bounds,
		zoom
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
	} = $props();

	const MIN_ZOOM = 14;
	const FETCH_PADDING_RATIO = 0.5;
	const STATION_LAYER_ID = 'rentals-stations';
	const VEHICLE_SOURCE_PREFIX = 'rentals-vehicles';
	const VEHICLE_CLUSTER_RADIUS = 50;
	const ZONE_LAYER_ID = 'rentals-zones';
	const ZONE_OUTLINE_LAYER_ID = 'rentals-zones-outline';

	const formFactors = Object.keys(formFactorAssets) as RentalFormFactor[];

	type RentalsPayload = Awaited<ReturnType<typeof rentals>>['data'];

	type DisplayFilter = {
		providerId: string;
		providerName: string;
		formFactor: RentalFormFactor;
	};

	let rentalsData = $state<RentalsPayload | null>(null);
	let loadedBounds = $state<maplibregl.LngLatBounds | null>(null);
	let requestToken = 0;
	let displayFilter = $state<DisplayFilter | null>(null);

	type StationFeatureProperties = {
		icon: string;
		available: number;
		providerId: string;
		stationId: string;
	};

	type VehicleFeatureProperties = {
		icon: string;
		providerId: string;
		vehicleId: string;
	};

	type VehicleCollections = Record<
		RentalFormFactor,
		FeatureCollection<Point, VehicleFeatureProperties>
	>;

	type ZoneFeatureProperties = {
		zoneIndex: number;
		providerId: string;
		z: number;
		rideEndAllowed: boolean;
		rideThroughAllowed: boolean;
	};

	const vehicleLayerConfigs = formFactors.map((formFactor) => {
		const slug = formFactor.toLowerCase();
		return {
			formFactor,
			sourceId: `${VEHICLE_SOURCE_PREFIX}-${slug}`,
			clusterLayerId: `${VEHICLE_SOURCE_PREFIX}-${slug}-cluster`,
			pointLayerId: `${VEHICLE_SOURCE_PREFIX}-${slug}-point`
		};
	});

	const buildProviderOptions = (providers: RentalProvider[]): DisplayFilter[] =>
		providers
			.flatMap((provider) =>
				provider.formFactors.map((formFactor) => ({
					providerId: provider.id,
					providerName: provider.name,
					formFactor
				}))
			)
			.sort(
				(a, b) =>
					a.providerName.localeCompare(b.providerName, undefined, { sensitivity: 'base' }) ||
					a.formFactor.localeCompare(b.formFactor, undefined, { sensitivity: 'base' })
			);

	let providerOptions = $derived(rentalsData ? buildProviderOptions(rentalsData.providers) : []);

	const isSameFilter = (a: DisplayFilter | null, b: DisplayFilter | null) =>
		a?.providerId === b?.providerId && a?.formFactor === b?.formFactor;

	const toggleFilter = (option: DisplayFilter) => {
		displayFilter = isSameFilter(displayFilter, option) ? null : option;
	};

	const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

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

	const fetchRentals = async (mapBounds: maplibregl.LngLatBounds) => {
		const expandedBounds = expandBounds(mapBounds);
		const max = lngLatToStr(expandedBounds.getNorthWest());
		const min = lngLatToStr(expandedBounds.getSouthEast());
		const token = ++requestToken;
		console.debug('[Rentals] requesting rentals', { min, max });

		const { data, error } = await rentals({ query: { max, min } });
		if (token !== requestToken) {
			return;
		}
		if (error) {
			console.error('[Rentals] rentals request failed', error);
			return;
		}

		rentalsData = data;
		loadedBounds = expandedBounds;
		const current = data;
		console.debug('[Rentals] received rentals', {
			providers: current.providers.length,
			stations: current.stations.length,
			vehicles: current.vehicles.length,
			zones: current.zones.length
		});
	};

	$effect(() => {
		if (!map || !bounds || zoom <= MIN_ZOOM) {
			rentalsData = null;
			loadedBounds = null;
			requestToken += 1;
			return;
		}

		const mapBounds = maplibregl.LngLatBounds.convert(bounds);
		if (boundsContain(loadedBounds, mapBounds)) {
			return;
		}

		void fetchRentals(mapBounds);
	});

	$effect(() => {
		if (displayFilter && !providerOptions.some((option) => isSameFilter(option, displayFilter))) {
			displayFilter = null;
		}
	});

	onDestroy(() => {
		requestToken += 1;
	});

	const stationAvailability = (
		station: RentalStation,
		formFactorByVehicleType: Map<string, RentalFormFactor> | null,
		filter: DisplayFilter | null
	) => {
		if (!filter || !formFactorByVehicleType) {
			return station.numVehiclesAvailable;
		}
		return Object.entries(station.vehicleTypesAvailable).reduce((sum, [typeId, count]) => {
			return formFactorByVehicleType.get(typeId) === filter.formFactor ? sum + count : sum;
		}, 0);
	};

	let stationFeatures = $derived.by((): FeatureCollection<Point, StationFeatureProperties> => {
		if (!rentalsData) {
			return { type: 'FeatureCollection', features: [] };
		}
		const filter = displayFilter;
		const stations = filter
			? rentalsData.stations.filter(
					(station) =>
						station.providerId === filter.providerId &&
						station.formFactors.includes(filter.formFactor)
				)
			: rentalsData.stations;
		const formFactorByVehicleType = filter
			? new Map(
					rentalsData.providers
						.find((p) => p.id === filter.providerId)
						?.vehicleTypes.map((type) => [type.id, type.formFactor]) ?? []
				)
			: null;
		return {
			type: 'FeatureCollection',
			features: stations.map((station) => {
				return {
					type: 'Feature',
					geometry: {
						type: 'Point',
						coordinates: [station.lon, station.lat]
					},
					properties: {
						icon: formFactorAssets[
							filter ? filter.formFactor : (station.formFactors[0] ?? DEFAULT_FORM_FACTOR)
						].station,
						available: stationAvailability(station, formFactorByVehicleType, filter),
						providerId: station.providerId,
						stationId: station.id
					}
				};
			})
		};
	});

	let vehicleCollections = $derived.by((): VehicleCollections => {
		const collections = formFactors.reduce((acc, formFactor) => {
			acc[formFactor] = { type: 'FeatureCollection', features: [] };
			return acc;
		}, {} as VehicleCollections);
		if (!rentalsData) {
			return collections;
		}
		const filter = displayFilter;
		const vehicles = filter
			? rentalsData.vehicles.filter(
					(vehicle) =>
						vehicle.providerId === filter.providerId && vehicle.formFactor === filter.formFactor
				)
			: rentalsData.vehicles;
		vehicles
			.filter((vehicle) => !vehicle.stationId)
			.forEach((vehicle) => {
				const collection = collections[vehicle.formFactor];
				collection.features.push({
					type: 'Feature',
					geometry: {
						type: 'Point',
						coordinates: [vehicle.lon, vehicle.lat]
					},
					properties: {
						icon: formFactorAssets[vehicle.formFactor].vehicle,
						providerId: vehicle.providerId,
						vehicleId: vehicle.id
					}
				});
			});
		return collections;
	});

	const buildZoneGeometry = (zone: RentalZone): GeoMultiPolygon => {
		return {
			type: 'MultiPolygon',
			coordinates: zone.area.map((polygon) =>
				polygon.map((encoded) =>
					polyline
						.decode(encoded.points, encoded.precision)
						.map(([lat, lng]) => [lng, lat] as Position)
				)
			)
		};
	};

	const findMatchingRule = (
		zone: RentalZone,
		provider: RentalProvider,
		formFactor: RentalFormFactor
	) => {
		return zone.rules.find((rule) => {
			if (!rule.vehicleTypeIdxs.length) {
				return true;
			}
			return rule.vehicleTypeIdxs.some(
				(idx) => provider.vehicleTypes[idx]?.formFactor === formFactor
			);
		});
	};

	let zoneFeatures = $derived.by(
		(): FeatureCollection<GeoPolygon | GeoMultiPolygon, ZoneFeatureProperties> => {
			const features: Array<Feature<GeoPolygon | GeoMultiPolygon, ZoneFeatureProperties>> = [];
			const provider = rentalsData?.providers?.find(
				(candidate) => candidate.id === displayFilter?.providerId
			);

			if (!rentalsData || !displayFilter || !provider) {
				return { type: 'FeatureCollection', features };
			}

			for (let i = 0; i < rentalsData.zones.length; ++i) {
				const zone = rentalsData.zones[i];
				if (zone.providerId !== displayFilter.providerId) {
					continue;
				}
				const rule = findMatchingRule(zone, provider, displayFilter.formFactor);
				if (!rule) {
					continue;
				}
				features.push({
					type: 'Feature',
					geometry: buildZoneGeometry(zone),
					properties: {
						zoneIndex: i,
						providerId: zone.providerId,
						z: zone.z,
						rideEndAllowed: rule.rideEndAllowed && !rule.stationParking,
						rideThroughAllowed: rule.rideThroughAllowed
					}
				});
			}
			return {
				type: 'FeatureCollection',
				features
			};
		}
	);

	const createScopedId = (providerId: string, entityId: string) => `${providerId}::${entityId}`;

	let providerById = $derived.by(
		(): Map<string, RentalProvider> => new Map(rentalsData?.providers.map((p) => [p.id, p]) ?? [])
	);

	let stationById = $derived.by(
		(): Map<string, RentalStation> =>
			new Map(rentalsData?.stations.map((s) => [createScopedId(s.providerId, s.id), s]) ?? [])
	);

	let vehicleById = $derived.by(
		(): Map<string, RentalVehicle> =>
			new Map(rentalsData?.vehicles.map((v) => [createScopedId(v.providerId, v.id), v]) ?? [])
	);

	const lookupStation = (properties: StationFeatureProperties) => {
		const key = createScopedId(properties.providerId, properties.stationId);
		return {
			key,
			provider: providerById.get(properties.providerId),
			station: stationById.get(key)
		};
	};

	const lookupVehicle = (properties: VehicleFeatureProperties) => {
		const key = createScopedId(properties.providerId, properties.vehicleId);
		return {
			key,
			provider: providerById.get(properties.providerId),
			vehicle: vehicleById.get(key)
		};
	};

	const lookupZone = (properties: ZoneFeatureProperties) => {
		const zone = rentalsData?.zones[properties.zoneIndex];
		return {
			key: String(properties.zoneIndex),
			provider: providerById.get(properties.providerId),
			zone,
			rideThroughAllowed: properties.rideThroughAllowed,
			rideEndAllowed: properties.rideEndAllowed
		};
	};

	const createStationContent = (
		provider: RentalProvider,
		station: RentalStation,
		showActions: boolean
	) => {
		const container = document.createElement('div');
		const component = mount(StationPopup, {
			target: container,
			props: { provider, station, showActions }
		});
		flushSync();
		return {
			element: container,
			destroy: () => {
				unmount(component);
			}
		};
	};

	const createVehicleContent = (
		provider: RentalProvider,
		vehicle: RentalVehicle,
		showActions: boolean
	) => {
		const container = document.createElement('div');
		const component = mount(VehiclePopup, {
			target: container,
			props: { provider, vehicle, showActions }
		});
		flushSync();
		return {
			element: container,
			destroy: () => {
				unmount(component);
			}
		};
	};

	const createZoneContent = (
		provider: RentalProvider,
		zone: RentalZone,
		rideThroughAllowed: boolean,
		rideEndAllowed: boolean
	) => {
		const container = document.createElement('div');
		const component = mount(ZonePopup, {
			target: container,
			props: { provider, zone, rideThroughAllowed, rideEndAllowed }
		});
		flushSync();
		return {
			element: container,
			destroy: () => {
				unmount(component);
			}
		};
	};

	let tooltipPopup: maplibregl.Popup | null = null;
	let detailPopup: maplibregl.Popup | null = null;
	let activeTooltipKey: string | null = null;
	let activePopupKey: string | null = null;
	let lastHoverCanvas: HTMLCanvasElement | null = null;
	let tooltipContentDestroy: (() => void) | null = null;
	let popupContentDestroy: (() => void) | null = null;

	const ensureTooltipPopup = () => {
		if (!tooltipPopup) {
			tooltipPopup = new maplibregl.Popup({
				closeButton: false,
				closeOnClick: false,
				offset: 12
			});
			tooltipPopup.on('close', () => {
				tooltipContentDestroy?.();
				tooltipContentDestroy = null;
				activeTooltipKey = null;
			});
		}
		return tooltipPopup;
	};

	const ensureDetailPopup = () => {
		if (!detailPopup) {
			detailPopup = new maplibregl.Popup({
				closeButton: true,
				closeOnClick: true,
				offset: 12
			});
			detailPopup.on('close', () => {
				popupContentDestroy?.();
				popupContentDestroy = null;
				activePopupKey = null;
			});
		}
		return detailPopup;
	};

	const hideTooltip = () => {
		tooltipContentDestroy?.();
		tooltipContentDestroy = null;
		if (tooltipPopup) {
			tooltipPopup.remove();
		}
		activeTooltipKey = null;
	};

	const hidePopup = () => {
		popupContentDestroy?.();
		popupContentDestroy = null;
		if (detailPopup) {
			detailPopup.remove();
		}
		activePopupKey = null;
	};

	const resetHoverCursor = () => {
		if (lastHoverCanvas) {
			lastHoverCanvas.style.cursor = '';
			lastHoverCanvas = null;
		}
	};

	const showTooltip = (
		targetMap: maplibregl.Map,
		lngLat: maplibregl.LngLatLike,
		key: string,
		createContent: () => { element: HTMLElement; destroy: () => void }
	) => {
		const popup = ensureTooltipPopup();
		if (activeTooltipKey !== key) {
			tooltipContentDestroy?.();
			const { element, destroy } = createContent();
			tooltipContentDestroy = destroy;
			popup.setDOMContent(element);
			activeTooltipKey = key;
		}
		popup.setLngLat(lngLat);
		if (!popup.isOpen()) {
			popup.addTo(targetMap);
		}
	};

	const showPopup = (
		targetMap: maplibregl.Map,
		lngLat: maplibregl.LngLatLike,
		key: string,
		createContent: () => { element: HTMLElement; destroy: () => void }
	) => {
		const popup = ensureDetailPopup();
		popupContentDestroy?.();
		const { element, destroy } = createContent();
		popupContentDestroy = destroy;
		popup.setDOMContent(element);
		popup.setLngLat(lngLat);
		if (!popup.isOpen()) {
			popup.addTo(targetMap);
		}
		activePopupKey = key;
	};

	const createMouseMoveHandler =
		<T,>(
			lookup: (properties: T) => {
				key: string;
				provider: RentalProvider | undefined;
				station?: RentalStation;
				vehicle?: RentalVehicle;
			},
			createContent: (
				provider: RentalProvider,
				entity: RentalStation | RentalVehicle
			) => { element: HTMLElement; destroy: () => void }
		) =>
		(event: MapLayerMouseEvent, mapInstance: maplibregl.Map) => {
			const feature = event.features?.[0];
			if (!mapInstance || !feature) {
				hideTooltip();
				resetHoverCursor();
				return;
			}
			const result = lookup(feature.properties as T);
			const entity = result.station || result.vehicle;
			if (!entity || !result.provider) {
				hideTooltip();
				resetHoverCursor();
				return;
			}
			lastHoverCanvas = mapInstance.getCanvas();
			lastHoverCanvas.style.cursor = 'pointer';
			if (activePopupKey === result.key) {
				hideTooltip();
				return;
			}
			showTooltip(mapInstance, event.lngLat, result.key, () =>
				createContent(result.provider!, entity)
			);
		};

	const createMouseLeaveHandler =
		() => (_event: MapLayerMouseEvent, _mapInstance: maplibregl.Map) => {
			resetHoverCursor();
			hideTooltip();
		};

	const createClickHandler =
		<T,>(
			lookup: (properties: T) => {
				key: string;
				provider: RentalProvider | undefined;
				station?: RentalStation;
				vehicle?: RentalVehicle;
			},
			createContent: (
				provider: RentalProvider,
				entity: RentalStation | RentalVehicle
			) => { element: HTMLElement; destroy: () => void }
		) =>
		(event: MapLayerMouseEvent, mapInstance: maplibregl.Map) => {
			const feature = event.features?.[0];
			if (!mapInstance || !feature) {
				hidePopup();
				return;
			}
			const result = lookup(feature.properties as T);
			const entity = result.station || result.vehicle;
			if (!entity || !result.provider) {
				hidePopup();
				return;
			}
			hideTooltip();
			hidePopup();
			showPopup(mapInstance, event.lngLat, result.key, () =>
				createContent(result.provider!, entity)
			);
		};

	const handleStationMouseMove = createMouseMoveHandler(lookupStation, (provider, entity) =>
		createStationContent(provider, entity as RentalStation, true)
	);
	const handleStationMouseLeave = createMouseLeaveHandler();
	const handleStationClick = createClickHandler(lookupStation, (provider, entity) =>
		createStationContent(provider, entity as RentalStation, true)
	);

	const handleVehicleMouseMove = createMouseMoveHandler(lookupVehicle, (provider, entity) =>
		createVehicleContent(provider, entity as RentalVehicle, true)
	);
	const handleVehicleMouseLeave = createMouseLeaveHandler();
	const handleVehicleClick = createClickHandler(lookupVehicle, (provider, entity) =>
		createVehicleContent(provider, entity as RentalVehicle, true)
	);

	const handleZoneClick = (event: MapLayerMouseEvent, mapInstance: maplibregl.Map) => {
		if (!mapInstance) {
			return;
		}

		// Check if there's a station or vehicle at this location (they should take priority)
		const vehiclePointLayers = vehicleLayerConfigs.map((c) => c.pointLayerId);
		const priorityLayers = [STATION_LAYER_ID, ...vehiclePointLayers];
		const priorityFeatures = mapInstance.queryRenderedFeatures(event.point, {
			layers: priorityLayers
		});
		if (priorityFeatures.length > 0) {
			// Let station/vehicle click handlers handle this
			return;
		}

		// event.features is already sorted by z-order (topmost first)
		const feature = event.features?.[0];
		if (!feature) {
			hidePopup();
			return;
		}

		const result = lookupZone(feature.properties as ZoneFeatureProperties);
		if (!result.zone || !result.provider) {
			hidePopup();
			return;
		}
		hideTooltip();
		hidePopup();
		showPopup(mapInstance, event.lngLat, result.key, () =>
			createZoneContent(
				result.provider!,
				result.zone!,
				result.rideThroughAllowed,
				result.rideEndAllowed
			)
		);
	};

	$effect(() => {
		if (!rentalsData) {
			hideTooltip();
			hidePopup();
			resetHoverCursor();
			return;
		}
		if (
			activeTooltipKey &&
			!stationById.has(activeTooltipKey) &&
			!vehicleById.has(activeTooltipKey)
		) {
			hideTooltip();
		}
		if (activePopupKey && !stationById.has(activePopupKey) && !vehicleById.has(activePopupKey)) {
			hidePopup();
		}
	});

	onDestroy(() => {
		hideTooltip();
		hidePopup();
		resetHoverCursor();
		tooltipPopup = null;
		detailPopup = null;
	});
</script>

{#if zoneFeatures.features.length > 0}
	{@const beforeLayerId = vehicleLayerConfigs[0]?.pointLayerId ?? STATION_LAYER_ID}
	<GeoJSON id={ZONE_LAYER_ID} data={zoneFeatures}>
		<Layer
			id={ZONE_LAYER_ID}
			{beforeLayerId}
			type="fill"
			filter={['literal', true]}
			layout={{ 'fill-sort-key': ['get', 'z'] }}
			onclick={handleZoneClick}
			paint={{
				'fill-color': [
					'case',
					['==', ['get', 'rideEndAllowed'], true],
					'#22c55e',
					['==', ['get', 'rideThroughAllowed'], false],
					'#ef4444',
					'#f97316'
				],
				'fill-opacity': 0.3
			}}
		/>
		<Layer
			id={ZONE_OUTLINE_LAYER_ID}
			{beforeLayerId}
			type="line"
			filter={['literal', true]}
			layout={{}}
			paint={{
				'line-color': [
					'case',
					['==', ['get', 'rideEndAllowed'], true],
					'#15803d',
					['==', ['get', 'rideThroughAllowed'], false],
					'#b91c1c',
					'#c2410c'
				],
				'line-width': 3,
				'line-opacity': 0.9
			}}
		/>
	</GeoJSON>
{/if}

{#if stationFeatures.features.length > 0}
	<GeoJSON id={STATION_LAYER_ID} data={stationFeatures}>
		<Layer
			id={STATION_LAYER_ID}
			beforeLayerId={undefined}
			type="symbol"
			filter={['literal', true]}
			layout={{
				'icon-image': ['get', 'icon'],
				'icon-size': zoomScaledIconSize,
				'icon-allow-overlap': true,
				'icon-ignore-placement': true,
				'text-field': ['to-string', ['get', 'available']],
				'text-allow-overlap': true,
				'text-ignore-placement': true,
				'text-anchor': 'center',
				'text-offset': zoomScaledTextOffset,
				'text-size': zoomScaledTextSizeMedium,
				'text-font': ['Noto Sans Display Regular']
			}}
			onmousemove={handleStationMouseMove}
			onmouseleave={handleStationMouseLeave}
			onclick={handleStationClick}
			paint={{
				'icon-opacity': 1,
				'text-color': '#1e293b',
				'text-halo-color': '#ffffff',
				'text-halo-width': 1.5
			}}
		/>
	</GeoJSON>
{/if}

{#each vehicleLayerConfigs as config (config.sourceId)}
	<GeoJSON
		id={config.sourceId}
		data={vehicleCollections[config.formFactor]}
		options={{ cluster: true, clusterRadius: VEHICLE_CLUSTER_RADIUS }}
	>
		<Layer
			id={config.clusterLayerId}
			beforeLayerId={STATION_LAYER_ID}
			type="symbol"
			filter={['has', 'point_count']}
			layout={{
				'icon-image': formFactorAssets[config.formFactor].cluster,
				'icon-size': zoomScaledIconSize,
				'icon-allow-overlap': true,
				'icon-ignore-placement': true,
				'text-field': ['to-string', ['get', 'point_count']],
				'text-allow-overlap': true,
				'text-ignore-placement': true,
				'text-anchor': 'center',
				'text-offset': zoomScaledTextOffset,
				'text-size': zoomScaledTextSizeSmall,
				'text-font': ['Noto Sans Display Regular']
			}}
			paint={{
				'icon-opacity': 1,
				'text-color': '#1e293b',
				'text-halo-color': '#ffffff',
				'text-halo-width': 1.5
			}}
		/>
		<Layer
			id={config.pointLayerId}
			beforeLayerId={STATION_LAYER_ID}
			type="symbol"
			filter={['!', ['has', 'point_count']]}
			layout={{
				'icon-image': formFactorAssets[config.formFactor].vehicle,
				'icon-size': zoomScaledIconSize,
				'icon-allow-overlap': true,
				'icon-ignore-placement': true
			}}
			onmousemove={handleVehicleMouseMove}
			onmouseleave={handleVehicleMouseLeave}
			onclick={handleVehicleClick}
			paint={{}}
		/>
	</GeoJSON>
{/each}

{#if providerOptions.length > 0}
	<Control position="bottom-right" class="mb-5">
		<div class="flex flex-col items-end space-y-2">
			{#each providerOptions as option (`${option.providerId}::${option.formFactor}`)}
				{@const active = isSameFilter(displayFilter, option)}
				<button
					type="button"
					title={`${option.providerName} (${formFactorAssets[option.formFactor].label})`}
					class={cn(
						'inline-flex max-w-sm items-center gap-2 rounded-md border-2 px-3 py-1.5 text-sm font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500',
						active
							? 'border-blue-600 bg-accent text-accent-foreground'
							: 'border-muted bg-popover text-foreground hover:bg-accent hover:text-accent-foreground'
					)}
					onclick={() => toggleFilter(option)}
					aria-pressed={active}
				>
					<span class="truncate">{option.providerName}</span>
					<span class="flex items-center gap-1 text-xs font-medium">
						<svg class="h-4 w-4 fill-current" aria-hidden="true" focusable="false">
							<use href={`#${formFactorAssets[option.formFactor].svg}`} />
						</svg>
					</span>
				</button>
			{/each}
		</div>
	</Control>
{/if}
