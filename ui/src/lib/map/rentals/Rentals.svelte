<script lang="ts">
	import { browser } from '$app/environment';
	import {
		rentals,
		type RentalFormFactor,
		type RentalProviderGroup,
		type RentalProvider,
		type RentalStation,
		type RentalVehicle,
		type RentalZone,
		type MultiPolygon
	} from '@motis-project/motis-client';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import Control from '$lib/map/Control.svelte';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import Layer from '$lib/map/Layer.svelte';
	import {
		DEFAULT_FORM_FACTOR,
		ICON_TYPES,
		formFactorAssets,
		colorizeIcon,
		getIconBaseName,
		getIconUrl,
		getIconDimensions,
		type IconType
	} from '$lib/map/rentals/assets';
	import { cn } from '$lib/utils';
	import polyline from '@mapbox/polyline';
	import maplibregl from 'maplibre-gl';
	import type { GeoJSONSource, MapLayerMouseEvent } from 'maplibre-gl';
	import { mount, onDestroy, unmount } from 'svelte';
	import type { FeatureCollection, Point, Position } from 'geojson';
	import RBush from 'rbush';
	import { point } from '@turf/helpers';
	import { booleanPointInPolygon } from '@turf/boolean-point-in-polygon';
	import StationPopup from '$lib/map/rentals/StationPopup.svelte';
	import VehiclePopup from '$lib/map/rentals/VehiclePopup.svelte';
	import ZonePopup from '$lib/map/rentals/ZonePopup.svelte';
	import {
		DEFAULT_COLOR,
		zoomScaledIconSize,
		zoomScaledTextOffset,
		zoomScaledTextSizeMedium,
		zoomScaledTextSizeSmall
	} from './style';
	import ZoneLayer from './ZoneLayer.svelte';
	import type {
		RentalZoneFeature,
		RentalZoneFeatureCollection,
		RentalZoneFeatureProperties
	} from './zone-types';
	import { SvelteSet } from 'svelte/reactivity';

	let zoneLayerRef = $state<ZoneLayer | null>(null);

	let {
		map,
		bounds,
		zoom,
		theme,
		debug = false
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		theme: 'light' | 'dark';
		debug?: boolean;
	} = $props();

	const MIN_ZOOM = 11;
	const FETCH_PADDING_RATIO = 0.5;
	const STATION_SOURCE_ID = 'rentals-stations';
	const STATION_ICON_LAYER_ID = 'rentals-stations-icons';
	const VEHICLE_SOURCE_PREFIX = 'rentals-vehicles';
	const VEHICLE_CLUSTER_RADIUS = 50;
	const ZONE_LAYER_ID = 'rentals-zones';

	const formFactors = Object.keys(formFactorAssets) as RentalFormFactor[];

	const getIconId = (formFactor: RentalFormFactor, type: IconType, color: string) =>
		`${getIconBaseName(formFactor, type)}-${color}`;

	type RentalsPayload = Awaited<ReturnType<typeof rentals>>['data'];
	type RentalsPayloadData = NonNullable<RentalsPayload>;

	type DisplayFilter = {
		providerGroupId: string;
		providerGroupName: string;
		formFactor: RentalFormFactor;
		color: string;
	};

	let rentalsData = $state<RentalsPayload | null>(null);
	let loadedBounds = $state<maplibregl.LngLatBounds | null>(null);
	let requestToken = 0;
	let displayFilter = $state<DisplayFilter | null>(null);
	let iconsReady = $state(false);
	let iconRequestToken = 0;
	let activeIconIds = new SvelteSet<string>();

	type StationFeatureProperties = {
		icon: string;
		available: number;
		providerId: string;
		stationId: string;
		color: string;
	};

	type VehicleFeatureProperties = {
		icon: string;
		clusterIcon: string;
		providerId: string;
		vehicleId: string;
		color: string;
	};

	type VehicleCollections = Record<
		RentalFormFactor,
		FeatureCollection<Point, VehicleFeatureProperties>
	>;

	const vehicleLayerConfigs = formFactors.map((formFactor) => {
		const slug = formFactor.toLowerCase();
		return {
			formFactor,
			sourceId: `${VEHICLE_SOURCE_PREFIX}-${slug}`,
			clusterLayerId: `${VEHICLE_SOURCE_PREFIX}-${slug}-cluster`,
			pointLayerId: `${VEHICLE_SOURCE_PREFIX}-${slug}-point`
		};
	});

	const buildProviderGroupOptions = (providerGroups: RentalProviderGroup[]): DisplayFilter[] =>
		providerGroups
			.flatMap((group) =>
				group.formFactors.map((formFactor) => ({
					providerGroupId: group.id,
					providerGroupName: group.name,
					formFactor,
					color: group.color || DEFAULT_COLOR
				}))
			)
			.sort(
				(a, b) =>
					a.providerGroupName.localeCompare(b.providerGroupName, undefined, {
						sensitivity: 'base'
					}) || a.formFactor.localeCompare(b.formFactor, undefined, { sensitivity: 'base' })
			);

	let providerGroupOptions = $derived(
		rentalsData ? buildProviderGroupOptions(rentalsData.providerGroups) : []
	);

	let providerGroupColors = $derived.by(
		() =>
			new Map(
				(rentalsData?.providerGroups ?? []).map((group) => [group.id, group.color || DEFAULT_COLOR])
			)
	);

	let providerColors = $derived.by(
		() =>
			new Map(
				(rentalsData?.providers ?? []).map((provider) => [
					provider.id,
					provider.color || providerGroupColors.get(provider.groupId) || DEFAULT_COLOR
				])
			)
	);

	type VehicleTypeFormFactors = Record<string, RentalFormFactor>;

	let vehicleTypeFormFactorsByProviderId = $derived.by(
		(): Record<string, VehicleTypeFormFactors> => {
			if (!rentalsData) {
				return {};
			}
			return Object.fromEntries(
				rentalsData.providers.map((provider) => [
					provider.id,
					Object.fromEntries(provider.vehicleTypes.map((type) => [type.id, type.formFactor]))
				])
			);
		}
	);

	const isSameFilter = (a: DisplayFilter | null, b: DisplayFilter | null) =>
		a?.providerGroupId === b?.providerGroupId && a?.formFactor === b?.formFactor;

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

	const collectFormFactorsByColor = (data: RentalsPayloadData) => {
		/* eslint-disable svelte/prefer-svelte-reactivity */
		const formFactorsByColor = new Map<string, Set<RentalFormFactor>>();
		const providerColorById = new Map<string, string>();
		const providerGroupColorById = new Map<string, string>(
			data.providerGroups.map((group) => [group.id, group.color || DEFAULT_COLOR])
		);
		/* eslint-enable svelte/prefer-svelte-reactivity */

		const addFormFactor = (color: string, formFactor: RentalFormFactor) => {
			let formFactors = formFactorsByColor.get(color);
			if (!formFactors) {
				/* eslint-disable-next-line svelte/prefer-svelte-reactivity */
				formFactors = new Set<RentalFormFactor>();
				formFactorsByColor.set(color, formFactors);
			}
			formFactors.add(formFactor);
		};

		for (const provider of data.providers) {
			const color = provider.color || providerGroupColorById.get(provider.groupId) || DEFAULT_COLOR;
			providerColorById.set(provider.id, color);
			for (const formFactor of provider.formFactors) {
				addFormFactor(color, formFactor);
			}
		}
		for (const station of data.stations) {
			const color =
				providerColorById.get(station.providerId) ||
				providerGroupColorById.get(station.providerGroupId) ||
				DEFAULT_COLOR;
			if (station.formFactors.length === 0) {
				addFormFactor(color, DEFAULT_FORM_FACTOR);
			}
			for (const formFactor of station.formFactors) {
				addFormFactor(color, formFactor);
			}
		}
		for (const vehicle of data.vehicles) {
			const color =
				providerColorById.get(vehicle.providerId) ||
				providerGroupColorById.get(vehicle.providerGroupId) ||
				DEFAULT_COLOR;
			addFormFactor(color, vehicle.formFactor);
		}
		return formFactorsByColor;
	};

	const stationAvailability = (
		station: RentalStation,
		formFactorsByProvider: Record<string, VehicleTypeFormFactors>,
		filter: DisplayFilter | null
	) => {
		if (!filter) {
			return station.numVehiclesAvailable;
		}
		const formFactors = formFactorsByProvider[station.providerId];
		if (!formFactors) {
			return station.numVehiclesAvailable;
		}
		return Object.entries(station.vehicleTypesAvailable).reduce((sum, [typeId, count]) => {
			return formFactors[typeId] === filter.formFactor ? sum + count : sum;
		}, 0);
	};

	let stationFeatures = $derived.by((): FeatureCollection<Point, StationFeatureProperties> => {
		if (!rentalsData || !iconsReady) {
			return { type: 'FeatureCollection', features: [] };
		}
		const filter = displayFilter;
		const stations = filter
			? rentalsData.stations.filter(
					(station) =>
						station.providerGroupId === filter.providerGroupId &&
						station.formFactors.includes(filter.formFactor)
				)
			: ([] as RentalStation[]);
		return {
			type: 'FeatureCollection',
			features: stations.map((station) => {
				const color =
					providerColors.get(station.providerId) ||
					providerGroupColors.get(station.providerGroupId) ||
					DEFAULT_COLOR;
				const formFactor = filter
					? filter.formFactor
					: (station.formFactors[0] ?? DEFAULT_FORM_FACTOR);
				return {
					type: 'Feature',
					geometry: {
						type: 'Point',
						coordinates: [station.lon, station.lat]
					},
					properties: {
						icon: getIconId(formFactor, 'station', color),
						available: stationAvailability(station, vehicleTypeFormFactorsByProviderId, filter),
						providerId: station.providerId,
						stationId: station.id,
						color
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
		if (!rentalsData || !iconsReady) {
			return collections;
		}
		const filter = displayFilter;
		const vehicles = filter
			? rentalsData.vehicles.filter(
					(vehicle) =>
						vehicle.providerGroupId === filter.providerGroupId &&
						vehicle.formFactor === filter.formFactor
				)
			: ([] as RentalVehicle[]);
		vehicles
			.filter((vehicle) => !vehicle.stationId)
			.forEach((vehicle) => {
				const collection = collections[vehicle.formFactor];
				const color =
					providerColors.get(vehicle.providerId) ||
					providerGroupColors.get(vehicle.providerGroupId) ||
					DEFAULT_COLOR;
				collection.features.push({
					type: 'Feature',
					geometry: {
						type: 'Point',
						coordinates: [vehicle.lon, vehicle.lat]
					},
					properties: {
						icon: getIconId(vehicle.formFactor, 'vehicle', color),
						clusterIcon: getIconId(vehicle.formFactor, 'cluster', color),
						providerId: vehicle.providerId,
						vehicleId: vehicle.id,
						color
					}
				});
			});
		return collections;
	});

	const buildZoneGeometry = (mp: MultiPolygon): RentalZoneFeature['geometry'] => {
		return {
			type: 'MultiPolygon',
			coordinates: mp.map((polygon) =>
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

	let zoneFeatures = $derived.by((): RentalZoneFeatureCollection => {
		const data = rentalsData;
		const filter = displayFilter;

		if (!data || !filter) {
			return { type: 'FeatureCollection', features: [] };
		}

		const features: RentalZoneFeature[] = [];
		let maxGeofencingZ = 0;

		data.zones.forEach((zone, index) => {
			if (zone.providerGroupId !== filter.providerGroupId) {
				return;
			}
			const provider = data.providers.find((candidate) => candidate.id === zone.providerId);
			if (!provider) {
				return;
			}
			const rule = findMatchingRule(zone, provider, filter.formFactor);
			if (!rule) {
				return;
			}
			features.push({
				type: 'Feature',
				geometry: buildZoneGeometry(zone.area),
				properties: {
					zoneIndex: index,
					providerId: zone.providerId,
					z: zone.z,
					rideEndAllowed: rule.rideEndAllowed && !rule.stationParking,
					rideThroughAllowed: rule.rideThroughAllowed,
					stationArea: false
				}
			} satisfies RentalZoneFeature);
			if (zone.z > maxGeofencingZ) {
				maxGeofencingZ = zone.z;
			}
		});

		const stationAreaZ = maxGeofencingZ + 1;
		data.stations.forEach((station, index) => {
			if (
				!station.stationArea ||
				station.providerGroupId !== filter.providerGroupId ||
				!station.formFactors.includes(filter.formFactor)
			) {
				return;
			}
			features.push({
				type: 'Feature',
				geometry: buildZoneGeometry(station.stationArea),
				properties: {
					stationIndex: index,
					providerId: station.providerId,
					z: stationAreaZ,
					rideEndAllowed: true,
					rideThroughAllowed: true,
					stationArea: true
				}
			} satisfies RentalZoneFeature);
		});

		features.sort((a, b) => b.properties.z - a.properties.z);

		return {
			type: 'FeatureCollection',
			features
		};
	});

	let zoneRTree = $derived.by(() => {
		const data = rentalsData;
		const rtree = new RBush<{
			minX: number;
			minY: number;
			maxX: number;
			maxY: number;
			feature: RentalZoneFeature;
		}>();
		if (data && debug) {
			rtree.load(
				zoneFeatures.features.map((feature) => {
					const bbox =
						feature.properties.zoneIndex !== undefined
							? data.zones[feature.properties.zoneIndex].bbox
							: data.stations[feature.properties.stationIndex!].bbox;
					return {
						minX: bbox[0],
						minY: bbox[1],
						maxX: bbox[2],
						maxY: bbox[3],
						feature
					};
				})
			);
		}
		return rtree;
	});

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

	const lookupZoneOrStation = (properties: RentalZoneFeatureProperties) => {
		const zone =
			properties.zoneIndex !== undefined ? rentalsData?.zones[properties.zoneIndex] : undefined;
		const station =
			properties.stationIndex !== undefined
				? rentalsData?.stations[properties.stationIndex]
				: undefined;
		return {
			key:
				properties.zoneIndex !== undefined
					? `zone-${properties.zoneIndex}`
					: `station-${properties.stationIndex}`,
			provider: providerById.get(properties.providerId),
			zone,
			station,
			rideThroughAllowed: properties.rideThroughAllowed,
			rideEndAllowed: properties.rideEndAllowed,
			stationArea: properties.stationArea
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
			props: { provider, station, showActions, debug }
		});
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
			props: { provider, vehicle, showActions, debug }
		});
		return {
			element: container,
			destroy: () => {
				unmount(component);
			}
		};
	};

	const createZoneContent = (
		provider: RentalProvider,
		zone: RentalZone | undefined,
		station: RentalStation | undefined,
		rideThroughAllowed: boolean,
		rideEndAllowed: boolean,
		allZonesAtPoint: RentalZoneFeature[]
	) => {
		const container = document.createElement('div');
		const component = mount(ZonePopup, {
			target: container,
			props: {
				provider,
				zone,
				station,
				rideThroughAllowed,
				rideEndAllowed,
				debug,
				allZonesAtPoint,
				zoneData: rentalsData?.zones ?? [],
				stationData: rentalsData?.stations ?? []
			}
		});
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
				offset: 12,
				maxWidth: 'none'
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
				offset: 12,
				maxWidth: 'none'
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

	const handleClusterClick = async (event: MapLayerMouseEvent, mapInstance: maplibregl.Map) => {
		const feature = event.features?.[0];
		const clusterId = feature?.properties?.cluster_id;
		const sourceId = feature?.source;
		if (!mapInstance || !feature || typeof clusterId !== 'number' || !sourceId) {
			return;
		}
		const source = mapInstance.getSource(sourceId) as GeoJSONSource | undefined;
		if (!source) {
			return;
		}

		hideTooltip();
		hidePopup();
		mapInstance.getCanvas().style.cursor = '';

		const zoom = await source.getClusterExpansionZoom(clusterId);
		mapInstance.easeTo({
			center: (feature.geometry as Point).coordinates as [number, number],
			zoom
		});
	};

	const handleClusterMouseEnter = (_event: MapLayerMouseEvent, mapInstance: maplibregl.Map) => {
		mapInstance.getCanvas().style.cursor = 'pointer';
	};

	const handleClusterMouseLeave = (_event: MapLayerMouseEvent, mapInstance: maplibregl.Map) => {
		mapInstance.getCanvas().style.cursor = '';
	};

	const handleVehicleMouseMove = createMouseMoveHandler(lookupVehicle, (provider, entity) =>
		createVehicleContent(provider, entity as RentalVehicle, true)
	);
	const handleVehicleMouseLeave = createMouseLeaveHandler();
	const handleVehicleClick = createClickHandler(lookupVehicle, (provider, entity) =>
		createVehicleContent(provider, entity as RentalVehicle, true)
	);

	const handleZoneClick = (
		event: maplibregl.MapMouseEvent,
		mapInstance: maplibregl.Map,
		layer: ZoneLayer
	) => {
		if (!mapInstance) {
			return;
		}

		// Check if there's a station or vehicle at this location (they should take priority)
		const priorityLayers = [
			STATION_ICON_LAYER_ID,
			...vehicleLayerConfigs.map((c) => c.pointLayerId),
			...vehicleLayerConfigs.map((c) => c.clusterLayerId)
		];
		const priorityFeatures = mapInstance.queryRenderedFeatures(event.point, {
			layers: priorityLayers
		});
		if (priorityFeatures.length > 0) {
			return;
		}

		const feature = layer.pick(event.point);
		if (!feature) {
			hidePopup();
			return;
		}

		const result = lookupZoneOrStation(feature.properties);
		if ((!result.zone && !result.station) || !result.provider) {
			hidePopup();
			return;
		}

		const allZonesAtPoint = zoneRTree
			.search({
				minX: event.lngLat.lng,
				minY: event.lngLat.lat,
				maxX: event.lngLat.lng,
				maxY: event.lngLat.lat
			})
			.map((item) => item.feature)
			.filter((feature) =>
				booleanPointInPolygon(point([event.lngLat.lng, event.lngLat.lat]), feature)
			)
			.sort((a, b) => b.properties.z - a.properties.z);
		hideTooltip();
		hidePopup();
		showPopup(mapInstance, event.lngLat, result.key, () =>
			createZoneContent(
				result.provider!,
				result.zone,
				result.station,
				result.rideThroughAllowed,
				result.rideEndAllowed,
				allZonesAtPoint
			)
		);
	};

	$effect(() => {
		if (!map || !bounds || zoom <= MIN_ZOOM) {
			if (zoom <= MIN_ZOOM) {
				rentalsData = null;
				loadedBounds = null;
				requestToken += 1;
			}
			return;
		}

		const mapBounds = maplibregl.LngLatBounds.convert(bounds);
		if (boundsContain(loadedBounds, mapBounds)) {
			return;
		}

		void fetchRentals(mapBounds);
	});

	$effect(() => {
		if (
			displayFilter &&
			!providerGroupOptions.some((option) => isSameFilter(option, displayFilter))
		) {
			displayFilter = null;
		}
	});

	const clearMapIcons = (mapInstance: maplibregl.Map | undefined) => {
		if (!mapInstance || activeIconIds.size === 0) {
			return;
		}
		for (const id of activeIconIds) {
			if (mapInstance.hasImage(id)) {
				mapInstance.removeImage(id);
			}
		}
		activeIconIds = new SvelteSet<string>();
	};

	$effect(() => {
		const mapInstance = map;
		const data = rentalsData;
		if (!browser || !mapInstance) {
			iconsReady = false;
			iconRequestToken += 1;
			clearMapIcons(mapInstance);
			return;
		}

		if (!data) {
			iconsReady = false;
			iconRequestToken += 1;
			clearMapIcons(mapInstance);
			return;
		}

		const dataPayload = data as RentalsPayloadData;
		const formFactorsByColor = collectFormFactorsByColor(dataPayload);
		if (formFactorsByColor.size === 0) {
			iconRequestToken += 1;
			clearMapIcons(mapInstance);
			iconsReady = true;
			return;
		}

		const token = ++iconRequestToken;
		iconsReady = false;
		/* eslint-disable-next-line svelte/prefer-svelte-reactivity */
		const neededIds = new Set<string>();
		const tasks: Promise<void>[] = [];

		formFactorsByColor.forEach((formFactors, color) => {
			formFactors.forEach((formFactor) => {
				ICON_TYPES.forEach((type) => {
					const dimensions = getIconDimensions(type);
					const id = getIconId(formFactor, type, color);
					neededIds.add(id);
					if (!mapInstance.hasImage(id)) {
						tasks.push(
							(async () => {
								try {
									const image = await colorizeIcon(getIconUrl(formFactor, type), color, dimensions);
									if (token !== iconRequestToken || !mapInstance) {
										return;
									}
									if (!mapInstance.hasImage(id)) {
										try {
											mapInstance.addImage(id, image);
										} catch (addError) {
											console.error(`[Rentals] failed to register icon ${id}`, addError);
										}
									}
								} catch (error) {
									console.error(`[Rentals] failed to prepare icon ${id}`, error);
								}
							})()
						);
					}
				});
			});
		});

		void (async () => {
			await Promise.allSettled(tasks);
			if (token !== iconRequestToken || !mapInstance) {
				return;
			}

			const previousIds = Array.from(activeIconIds);
			for (const id of previousIds) {
				if (!neededIds.has(id) && mapInstance.hasImage(id)) {
					mapInstance.removeImage(id);
				}
			}

			const newActiveIds = new SvelteSet<string>();
			neededIds.forEach((id) => {
				if (mapInstance.hasImage(id)) {
					newActiveIds.add(id);
				}
			});
			activeIconIds = newActiveIds;
			iconsReady = neededIds.size === newActiveIds.size;
		})();
	});

	$effect(() => {
		const mapInstance = map;
		const layer = zoneLayerRef;
		const hasZones = zoneFeatures.features.length > 0;
		if (!mapInstance || !layer || !hasZones) {
			return;
		}

		const handler = (event: maplibregl.MapMouseEvent) => {
			handleZoneClick(event, mapInstance, layer);
		};
		mapInstance.on('click', handler);
		return () => {
			mapInstance.off('click', handler);
		};
	});

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
		requestToken += 1;
		hideTooltip();
		hidePopup();
		resetHoverCursor();
		clearMapIcons(map);
		tooltipPopup = null;
		detailPopup = null;
	});
</script>

{#if zoneFeatures.features.length > 0}
	{@const beforeLayerId = vehicleLayerConfigs[0]?.pointLayerId ?? STATION_ICON_LAYER_ID}
	<ZoneLayer
		bind:this={zoneLayerRef}
		id={ZONE_LAYER_ID}
		features={zoneFeatures.features}
		{beforeLayerId}
		opacity={theme === 'dark' ? 0.3 : 0.4}
	/>
{/if}

<GeoJSON id={STATION_SOURCE_ID} data={stationFeatures}>
	<Layer
		id={STATION_ICON_LAYER_ID}
		beforeLayerId=""
		type="symbol"
		filter={true}
		layout={{
			'icon-image': ['get', 'icon'],
			'icon-size': zoomScaledIconSize,
			'icon-allow-overlap': false,
			'icon-ignore-placement': true,
			'text-field': ['to-string', ['get', 'available']],
			'text-allow-overlap': false,
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
			'text-color': '#000'
		}}
	/>
</GeoJSON>

{#each vehicleLayerConfigs as config (config.sourceId)}
	<GeoJSON
		id={config.sourceId}
		data={vehicleCollections[config.formFactor]}
		options={{
			cluster: true,
			clusterRadius: VEHICLE_CLUSTER_RADIUS,
			clusterProperties: {
				color: ['coalesce', ['get', 'color']],
				icon: ['coalesce', ['get', 'icon']],
				clusterIcon: ['coalesce', ['get', 'clusterIcon']]
			}
		}}
	>
		<Layer
			id={config.clusterLayerId}
			beforeLayerId={STATION_ICON_LAYER_ID}
			type="symbol"
			filter={['has', 'point_count']}
			layout={{
				'icon-image': ['get', 'clusterIcon'],
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
				'text-color': '#000'
			}}
			onclick={handleClusterClick}
			onmouseenter={handleClusterMouseEnter}
			onmouseleave={handleClusterMouseLeave}
		/>
		<Layer
			id={config.pointLayerId}
			beforeLayerId={config.clusterLayerId}
			type="symbol"
			filter={['!', ['has', 'point_count']]}
			layout={{
				'icon-image': ['get', 'icon'],
				'icon-size': zoomScaledIconSize,
				'icon-allow-overlap': true,
				'icon-ignore-placement': true
			}}
			paint={{}}
			onmousemove={handleVehicleMouseMove}
			onmouseleave={handleVehicleMouseLeave}
			onclick={handleVehicleClick}
		/>
	</GeoJSON>
{/each}

{#if providerGroupOptions.length > 0}
	<Control position="top-right" class="mb-5">
		<div class="flex flex-col items-end space-y-2">
			{#each providerGroupOptions as option (`${option.providerGroupId}::${option.formFactor}`)}
				{@const active = isSameFilter(displayFilter, option)}
				<button
					type="button"
					title={`${option.providerGroupName} (${formFactorAssets[option.formFactor].label})`}
					class={cn(
						'inline-flex max-w-72 items-center gap-2 rounded-md border-2 px-3 py-1.5 text-sm font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500',
						active
							? 'border-blue-600 bg-accent text-accent-foreground'
							: 'border-muted bg-popover text-foreground hover:bg-accent hover:text-accent-foreground'
					)}
					onclick={() => toggleFilter(option)}
					aria-pressed={active}
				>
					<span class="truncate">{option.providerGroupName}</span>
					<span class="flex items-center gap-1 text-xs font-medium">
						<svg
							class="h-4 w-4 fill-current"
							aria-hidden="true"
							focusable="false"
							style={`color: ${option.color}`}
						>
							<use href={`#${formFactorAssets[option.formFactor].svg}`} />
						</svg>
					</span>
				</button>
			{/each}
		</div>
	</Control>
{/if}
