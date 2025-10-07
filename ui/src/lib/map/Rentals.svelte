<script lang="ts">
	import {
		rentals,
		type EncodedPolyline,
		type RentalFormFactor,
		type RentalProvider,
		type RentalStation,
		type RentalVehicle,
		type RentalZone
	} from '$lib/api/openapi';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import polyline from '@mapbox/polyline';
	import maplibregl from 'maplibre-gl';
	import type { ExpressionSpecification } from 'maplibre-gl';
	import { onDestroy } from 'svelte';
	import { t } from '$lib/i18n/translation';
	import type {
		FeatureCollection,
		Feature,
		MultiPolygon as GeoJSONMultiPolygon,
		Point,
		Polygon as GeoJSONPolygon,
		Position
	} from 'geojson';

	let {
		map,
		bounds,
		zoom,
		showZones
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		showZones: boolean;
	} = $props();

	const MIN_ZOOM = 14;
	const FETCH_PADDING_RATIO = 0.5;
	const ZONES_SOURCE_ID = 'rentals-zones';
	const ZONES_LAYER_ID = 'rentals-zones-layer';
	const STATIONS_SOURCE_ID = 'rentals-stations';
	const STATIONS_LAYER_ID = 'rentals-stations-layer';
	const VEHICLES_SOURCE_PREFIX = 'rentals-vehicles';
	const VEHICLES_LAYER_PREFIX = 'rentals-vehicles-layer';
	const VEHICLES_CLUSTER_LAYER_PREFIX = 'rentals-vehicles-cluster-layer';
	const DEFAULT_FORM_FACTOR = 'BICYCLE';

	let loadedBounds: maplibregl.LngLatBounds | null = null;
	let requestToken = 0;
	let activeMap: maplibregl.Map | undefined;
	let providerById = new Map<string, RentalProvider>();
	let stationById = new Map<string, RentalStation>();
	/* eslint-disable-next-line @typescript-eslint/no-unused-vars */
	let vehicleById = new Map<string, RentalVehicle>();
	let zones: RentalZone[] = [];

	const createScopedId = (providerId: string, entityId: string) => `${providerId}::${entityId}`;

	type StationFeatureProperties = {
		icon: string;
		icon_empty: string;
		available: number;
		provider_id: string;
		station_id: string;
		type: 'station';
	};

	type VehicleFeatureProperties = {
		icon: string;
		provider_id: string;
		vehicle_id: string;
		type: 'vehicle';
	};

	type VehicleFeatureCollection = FeatureCollection<Point, VehicleFeatureProperties>;
	type VehicleCollections = Record<string, VehicleFeatureCollection>;

	type ZoneFeatureProperties = {
		zone_index: number;
		provider_id: string;
		type: 'zone';
	};

	type ZoneFeatureCollection = FeatureCollection<
		GeoJSONPolygon | GeoJSONMultiPolygon,
		ZoneFeatureProperties
	>;

	const create_empty_vehicle_collection = (): VehicleFeatureCollection => ({
		type: 'FeatureCollection',
		features: [] as Feature<Point, VehicleFeatureProperties>[]
	});

	let stationData: FeatureCollection<Point, StationFeatureProperties> = {
		type: 'FeatureCollection',
		features: []
	};
	let vehicleDataByIcon: VehicleCollections;
	let zoneData: ZoneFeatureCollection = {
		type: 'FeatureCollection',
		features: []
	};

	const station_icon_by_form_factor: Record<RentalFormFactor, string> = {
		BICYCLE: 'bike_station',
		CARGO_BICYCLE: 'cargo_bike_station',
		CAR: 'car_station',
		MOPED: 'moped_station',
		SCOOTER_SEATED: 'scooter_station',
		SCOOTER_STANDING: 'scooter_station',
		OTHER: 'bike_station'
	};

	const station_icon_empty_by_form_factor: Record<RentalFormFactor, string> = {
		BICYCLE: 'bike_station_empty',
		CARGO_BICYCLE: 'cargo_bike_station_empty',
		CAR: 'car_station_empty',
		MOPED: 'moped_station_empty',
		SCOOTER_SEATED: 'scooter_station_empty',
		SCOOTER_STANDING: 'scooter_station_empty',
		OTHER: 'bike_station_empty'
	};

	const vehicle_icon_by_form_factor: Record<RentalFormFactor, string> = {
		BICYCLE: 'floating_bike',
		CARGO_BICYCLE: 'floating_cargo_bike',
		CAR: 'floating_car',
		MOPED: 'floating_moped',
		SCOOTER_SEATED: 'floating_scooter',
		SCOOTER_STANDING: 'floating_scooter',
		OTHER: 'floating_bike'
	};

	const vehicle_icons = Array.from(new Set(Object.values(vehicle_icon_by_form_factor)));

	type VehicleSourceConfig = {
		icon: string;
		source_id: string;
		cluster_layer_id: string;
		point_layer_id: string;
		cluster_icon: string;
	};

	const vehicle_source_configs: VehicleSourceConfig[] = vehicle_icons.map((icon) => ({
		icon,
		source_id: `${VEHICLES_SOURCE_PREFIX}-${icon}`,
		cluster_layer_id: `${VEHICLES_CLUSTER_LAYER_PREFIX}-${icon}`,
		point_layer_id: `${VEHICLES_LAYER_PREFIX}-${icon}`,
		cluster_icon: `${icon}_cluster`
	}));

	const vehicle_source_config_by_icon = new Map(
		vehicle_source_configs.map((config) => [config.icon, config] as const)
	);

	const vehicle_point_layer_ids = vehicle_source_configs.map((config) => config.point_layer_id);
	const vehicle_cluster_layer_ids = vehicle_source_configs.map((config) => config.cluster_layer_id);

	const create_empty_vehicle_collections = (): VehicleCollections =>
		vehicle_source_configs.reduce((acc, config) => {
			acc[config.icon] = create_empty_vehicle_collection();
			return acc;
		}, {} as VehicleCollections);

	vehicleDataByIcon = create_empty_vehicle_collections();

	const create_empty_zone_collection = (): ZoneFeatureCollection => ({
		type: 'FeatureCollection',
		features: []
	});

	const decodePolylinePositions = (encoded: EncodedPolyline): Position[] => {
		if (!encoded?.points || typeof encoded.precision !== 'number') {
			return [];
		}
		try {
			return polyline
				.decode(encoded.points, encoded.precision)
				.map(([lat, lng]) => [lng, lat] as Position);
		} catch (error) {
			console.error('Failed to decode rental zone polyline', error);
			return [];
		}
	};

	const ensureRingClosed = (ring: Position[]): Position[] => {
		if (ring.length === 0) {
			return ring;
		}
		const [firstLng, firstLat] = ring[0];
		const [lastLng, lastLat] = ring[ring.length - 1];
		if (firstLng === lastLng && firstLat === lastLat) {
			return ring;
		}
		return [...ring, ring[0]];
	};

	const buildZoneGeometry = (zone: RentalZone): GeoJSONPolygon | GeoJSONMultiPolygon | null => {
		const polygons = (zone.area ?? []).reduce<Position[][][]>((acc, polygon) => {
			const rings = polygon
				.map((encodedRing) => ensureRingClosed(decodePolylinePositions(encodedRing)))
				.filter((ring) => ring.length >= 4);
			if (rings.length > 0) {
				acc.push(rings);
			}
			return acc;
		}, []);
		if (polygons.length === 0) {
			return null;
		}
		if (polygons.length === 1) {
			return {
				type: 'Polygon',
				coordinates: polygons[0]
			};
		}
		return {
			type: 'MultiPolygon',
			coordinates: polygons
		};
	};

	const buildZoneFeatures = (zonesInput: RentalZone[]): ZoneFeatureCollection => {
		const features = zonesInput
			.map<Feature<GeoJSONPolygon | GeoJSONMultiPolygon, ZoneFeatureProperties> | null>(
				(zone, index) => {
					const geometry = buildZoneGeometry(zone);
					if (!geometry) {
						return null;
					}
					return {
						type: 'Feature',
						id: `zone:${index}`,
						geometry,
						properties: {
							zone_index: index,
							provider_id: zone.providerId,
							type: 'zone'
						}
					};
				}
			)
			.filter(
				(
					feature
				): feature is Feature<GeoJSONPolygon | GeoJSONMultiPolygon, ZoneFeatureProperties> =>
					feature !== null
			);
		return {
			type: 'FeatureCollection',
			features
		};
	};

	const zoom_scaled_icon_size: ExpressionSpecification = [
		'interpolate',
		['linear'],
		['zoom'],
		14,
		0.6,
		18,
		1
	];

	const zoom_scaled_text_size_small: ExpressionSpecification = [
		'interpolate',
		['linear'],
		['zoom'],
		14,
		6,
		18,
		10
	];

	const zoom_scaled_text_size_medium: ExpressionSpecification = [
		'interpolate',
		['linear'],
		['zoom'],
		14,
		7.2,
		18,
		12
	];

	const create_zoom_scaled_text_offset = (baseOffset: [number, number]): ExpressionSpecification =>
		[
			'interpolate',
			['linear'],
			['zoom'],
			14,
			['literal', [baseOffset[0] * 0.8, baseOffset[1] * 0.9]],
			18,
			['literal', baseOffset]
		] as unknown as ExpressionSpecification;

	const zoom_scaled_text_offset = create_zoom_scaled_text_offset([0.8, -1.25]);

	const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

	const expandBounds = (bounds: maplibregl.LngLatBounds) => {
		const sw = bounds.getSouthWest();
		const ne = bounds.getNorthEast();
		const width = ne.lng - sw.lng;
		const height = ne.lat - sw.lat;
		const padLng = width * FETCH_PADDING_RATIO;
		const padLat = height * FETCH_PADDING_RATIO;
		return new maplibregl.LngLatBounds(
			[clamp(sw.lng - padLng, -180, 180), clamp(sw.lat - padLat, -90, 90)],
			[clamp(ne.lng + padLng, -180, 180), clamp(ne.lat + padLat, -90, 90)]
		);
	};

	const containsBounds = (
		outer: maplibregl.LngLatBounds | null,
		inner: maplibregl.LngLatBounds
	) => {
		if (!outer) {
			return false;
		}
		return outer.contains(inner.getSouthWest()) && outer.contains(inner.getNorthEast());
	};

	const getStationIconName = (station: RentalStation) => {
		const formFactor = station.formFactors?.[0];
		return station_icon_by_form_factor[formFactor ?? DEFAULT_FORM_FACTOR];
	};

	const getStationEmptyIconName = (station: RentalStation) => {
		const formFactor = station.formFactors?.[0];
		return station_icon_empty_by_form_factor[formFactor ?? DEFAULT_FORM_FACTOR];
	};

	const getVehicleIconName = (vehicle: RentalVehicle) => {
		return vehicle_icon_by_form_factor[vehicle.formFactor];
	};

	const buildStationFeatures = (
		stations: RentalStation[]
	): FeatureCollection<Point, StationFeatureProperties> => {
		return {
			type: 'FeatureCollection',
			features: stations
				.map<Feature<Point, StationFeatureProperties> | null>((station) => {
					const available = station.numVehiclesAvailable ?? 0;
					const scopedId = createScopedId(station.providerId, station.id);
					return {
						type: 'Feature',
						id: `station:${scopedId}`,
						geometry: {
							type: 'Point',
							coordinates: [station.lon, station.lat]
						},
						properties: {
							icon: getStationIconName(station),
							icon_empty: getStationEmptyIconName(station),
							available,
							provider_id: station.providerId,
							station_id: station.id,
							type: 'station'
						}
					};
				})
				.filter((feature): feature is Feature<Point, StationFeatureProperties> => feature !== null)
		};
	};

	const buildVehicleFeatures = (vehicles: RentalVehicle[]): VehicleCollections => {
		const collections = create_empty_vehicle_collections();
		const fallbackConfig = vehicle_source_configs[0];
		if (!fallbackConfig) {
			return collections;
		}
		vehicles
			.filter((vehicle) => !vehicle.stationId)
			.forEach((vehicle) => {
				const iconName = getVehicleIconName(vehicle);
				const config = vehicle_source_config_by_icon.get(iconName) ?? fallbackConfig;
				const targetCollection = collections[config.icon];
				if (!targetCollection) {
					return;
				}
				const scopedId = createScopedId(vehicle.providerId, vehicle.id);
				targetCollection.features.push({
					type: 'Feature',
					id: `vehicle:${scopedId}`,
					geometry: {
						type: 'Point',
						coordinates: [vehicle.lon, vehicle.lat]
					},
					properties: {
						icon: config.icon,
						provider_id: vehicle.providerId,
						vehicle_id: vehicle.id,
						type: 'vehicle'
					}
				});
			});
		return collections;
	};

	const ensureSourcesAndLayers = (targetMap: maplibregl.Map | undefined) => {
		if (!targetMap || !targetMap.isStyleLoaded()) {
			return;
		}

		if (!targetMap.getSource(ZONES_SOURCE_ID)) {
			targetMap.addSource(ZONES_SOURCE_ID, {
				type: 'geojson',
				data: zoneData
			});
		}

		if (!targetMap.getLayer(ZONES_LAYER_ID)) {
			targetMap.addLayer({
				id: ZONES_LAYER_ID,
				type: 'fill',
				source: ZONES_SOURCE_ID,
				paint: {
					'fill-color': '#0ea5e9',
					'fill-opacity': 0.12,
					'fill-outline-color': '#0284c7'
				}
			});
		}

		for (const config of vehicle_source_configs) {
			const data = vehicleDataByIcon[config.icon] ?? create_empty_vehicle_collection();
			if (!targetMap.getSource(config.source_id)) {
				targetMap.addSource(config.source_id, {
					type: 'geojson',
					data,
					cluster: true,
					clusterRadius: 50
				});
			}
			if (!targetMap.getLayer(config.cluster_layer_id)) {
				targetMap.addLayer({
					id: config.cluster_layer_id,
					type: 'symbol',
					source: config.source_id,
					filter: ['has', 'point_count'],
					layout: {
						'icon-image': config.cluster_icon,
						'icon-size': zoom_scaled_icon_size,
						'icon-allow-overlap': true,
						'icon-ignore-placement': true,
						'text-field': ['to-string', ['get', 'point_count']],
						'text-allow-overlap': true,
						'text-ignore-placement': true,
						'text-anchor': 'center',
						'text-offset': zoom_scaled_text_offset,
						'text-size': zoom_scaled_text_size_small,
						'text-font': ['Noto Sans Display Regular']
					},
					paint: {
						'text-color': '#1e293b',
						'text-halo-color': '#ffffff',
						'text-halo-width': 1.5
					}
				});
			}
			if (!targetMap.getLayer(config.point_layer_id)) {
				targetMap.addLayer({
					id: config.point_layer_id,
					type: 'symbol',
					source: config.source_id,
					filter: ['!', ['has', 'point_count']],
					layout: {
						'icon-image': config.icon,
						'icon-size': zoom_scaled_icon_size,
						'icon-allow-overlap': true,
						'icon-ignore-placement': true
					}
				});
			}
		}

		if (!targetMap.getSource(STATIONS_SOURCE_ID)) {
			targetMap.addSource(STATIONS_SOURCE_ID, {
				type: 'geojson',
				data: stationData
			});
		}

		if (!targetMap.getLayer(STATIONS_LAYER_ID)) {
			targetMap.addLayer({
				id: STATIONS_LAYER_ID,
				type: 'symbol',
				source: STATIONS_SOURCE_ID,
				layout: {
					'icon-image': [
						'case',
						['>', ['get', 'available'], 0],
						['get', 'icon'],
						['get', 'icon_empty']
					],
					'icon-size': zoom_scaled_icon_size,
					'icon-allow-overlap': true,
					'icon-ignore-placement': true,
					'text-field': [
						'case',
						['>', ['get', 'available'], 0],
						['to-string', ['get', 'available']],
						''
					],
					'text-allow-overlap': true,
					'text-ignore-placement': true,
					'text-anchor': 'center',
					'text-offset': zoom_scaled_text_offset,
					'text-size': zoom_scaled_text_size_medium,
					'text-font': ['Noto Sans Display Regular']
				},
				paint: {
					'icon-opacity': ['case', ['>', ['get', 'available'], 0], 1, 0.7],
					'text-color': '#1e293b',
					'text-halo-color': '#ffffff',
					'text-halo-width': 1.5
				}
			});
		}
	};

	const removeSourcesAndLayers = (targetMap: maplibregl.Map | undefined) => {
		if (!targetMap) {
			return;
		}
		for (const config of vehicle_source_configs) {
			if (targetMap.getLayer(config.point_layer_id)) {
				targetMap.removeLayer(config.point_layer_id);
			}
			if (targetMap.getLayer(config.cluster_layer_id)) {
				targetMap.removeLayer(config.cluster_layer_id);
			}
			if (targetMap.getSource(config.source_id)) {
				targetMap.removeSource(config.source_id);
			}
		}
		if (targetMap.getLayer(STATIONS_LAYER_ID)) {
			targetMap.removeLayer(STATIONS_LAYER_ID);
		}
		if (targetMap.getSource(STATIONS_SOURCE_ID)) {
			targetMap.removeSource(STATIONS_SOURCE_ID);
		}
		if (targetMap.getLayer(ZONES_LAYER_ID)) {
			targetMap.removeLayer(ZONES_LAYER_ID);
		}
		if (targetMap.getSource(ZONES_SOURCE_ID)) {
			targetMap.removeSource(ZONES_SOURCE_ID);
		}
	};

	const setStationSourceData = (
		data: FeatureCollection<Point, StationFeatureProperties>,
		targetMap: maplibregl.Map | undefined
	) => {
		stationData = data;
		if (!targetMap) {
			return;
		}
		ensureSourcesAndLayers(targetMap);
		const source = targetMap.getSource(STATIONS_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
		source?.setData(data);
	};

	const setVehicleSourceData = (
		dataByIcon: VehicleCollections,
		targetMap: maplibregl.Map | undefined
	) => {
		const nextData = create_empty_vehicle_collections();
		for (const config of vehicle_source_configs) {
			nextData[config.icon] = dataByIcon?.[config.icon] ?? create_empty_vehicle_collection();
		}
		vehicleDataByIcon = nextData;
		if (!targetMap) {
			return;
		}
		ensureSourcesAndLayers(targetMap);
		for (const config of vehicle_source_configs) {
			const source = targetMap.getSource(config.source_id) as maplibregl.GeoJSONSource | undefined;
			source?.setData(vehicleDataByIcon[config.icon]);
		}
	};

	const setZoneSourceData = (
		data: ZoneFeatureCollection,
		targetMap: maplibregl.Map | undefined
	) => {
		zoneData = data;
		if (!targetMap) {
			return;
		}
		ensureSourcesAndLayers(targetMap);
		const source = targetMap.getSource(ZONES_SOURCE_ID) as maplibregl.GeoJSONSource | undefined;
		source?.setData(zoneData);
	};

	const styleListeners = new WeakMap<maplibregl.Map, () => void>();
	const pointerHandlers = new WeakMap<
		maplibregl.Map,
		{
			stationEnter: (event: maplibregl.MapLayerMouseEvent) => void;
			stationMove: (event: maplibregl.MapMouseEvent) => void;
			stationLeave: () => void;
			vehicleEnter: (event: maplibregl.MapLayerMouseEvent) => void;
			vehicleMove: (event: maplibregl.MapMouseEvent) => void;
			vehicleLeave: () => void;
			vehicleClusterEnter: () => void;
			vehicleClusterLeave: () => void;
			vehicleLayerIds: string[];
			vehicleClusterLayerIds: string[];
			zoneEnter: (event: maplibregl.MapLayerMouseEvent) => void;
			zoneLeave: () => void;
			zoneClick: (event: maplibregl.MapLayerMouseEvent) => void;
			zoneMapClick: (event: maplibregl.MapMouseEvent) => void;
			zoneLayerId: string;
		}
	>();
	const clusterClickHandlers = new WeakMap<
		maplibregl.Map,
		Map<string, (event: maplibregl.MapLayerMouseEvent) => void>
	>();
	const mapPopups = new WeakMap<maplibregl.Map, maplibregl.Popup>();

	const escapeHtml = (value: string) =>
		value
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;')
			.replace(/"/g, '&quot;');

	const formatTooltipHtml = (value: string | undefined) => {
		if (!value) {
			return '';
		}
		const lines = value
			.split('\n')
			.map((line) => line.trim())
			.filter((line) => line.length > 0)
			.map((line) => escapeHtml(line));
		return lines.join('<br>');
	};

	const createTooltipHtml = (...lines: Array<string | null | undefined>) => {
		const filtered = lines.map((line) => (line ?? '').trim()).filter((line) => line.length > 0);
		if (filtered.length === 0) {
			return '';
		}
		return formatTooltipHtml(filtered.join('\n'));
	};

	const getStringProperty = (
		properties: Record<string, unknown> | undefined,
		key: string
	): string | undefined => {
		if (!properties) {
			return undefined;
		}
		const value = properties[key];
		return typeof value === 'string' ? value : undefined;
	};

	const getNumberProperty = (
		properties: Record<string, unknown> | undefined,
		key: string
	): number | undefined => {
		if (!properties) {
			return undefined;
		}
		const value = properties[key];
		if (typeof value === 'number' && Number.isFinite(value)) {
			return value;
		}
		if (typeof value === 'string') {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : undefined;
		}
		return undefined;
	};

	const buildStationTooltipHtml = (feature: maplibregl.MapGeoJSONFeature | undefined) => {
		const properties = feature?.properties as Record<string, unknown> | undefined;
		const stationId = getStringProperty(properties, 'station_id');
		const providerId = getStringProperty(properties, 'provider_id');
		if (!stationId || !providerId) {
			return '';
		}
		const station = stationById.get(createScopedId(providerId, stationId));
		if (!station) {
			return '';
		}
		const providerName = providerById.get(providerId)?.name;
		return createTooltipHtml(
			station.name ?? undefined,
			providerName ? `${t.sharingProvider}: ${providerName}` : undefined
		);
	};

	const buildVehicleTooltipHtml = (feature: maplibregl.MapGeoJSONFeature | undefined) => {
		const properties = feature?.properties as Record<string, unknown> | undefined;
		const vehicleId = getStringProperty(properties, 'vehicle_id');
		const providerId = getStringProperty(properties, 'provider_id');
		if (!vehicleId || !providerId) {
			return '';
		}
		const providerName = providerById.get(providerId)?.name;
		if (!providerName) {
			return '';
		}
		return createTooltipHtml(`${t.sharingProvider}: ${providerName}`);
	};

	const buildZoneTooltipSection = (zone: RentalZone) => {
		const providerName = providerById.get(zone.providerId)?.name;
		const headerParts: string[] = [];
		if (zone.name) {
			headerParts.push(`<div>${escapeHtml(zone.name)}</div>`);
		}
		if (providerName) {
			headerParts.push(`<div>${escapeHtml(`${t.sharingProvider}: ${providerName}`)}</div>`);
		}
		const headerHtml = headerParts.join('');
		const restrictionsJson = JSON.stringify(zone.rules ?? [], null, 2);
		const restrictionsHtml = `<pre class="m-0 max-w-full overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs leading-snug bg-slate-50 p-2 rounded">${escapeHtml(restrictionsJson ?? '[]')}</pre>`;
		return `<div class="space-y-2.5">${headerHtml}${restrictionsHtml}</div>`;
	};

	const buildZoneTooltipHtml = (features: maplibregl.MapGeoJSONFeature[] | undefined) => {
		if (!features || features.length === 0) {
			return '';
		}
		const seen = new Set<number>();
		const sections: string[] = [];
		for (const feature of features) {
			const properties = feature?.properties as Record<string, unknown> | undefined;
			const zoneIndex = getNumberProperty(properties, 'zone_index');
			if (
				zoneIndex === undefined ||
				!Number.isInteger(zoneIndex) ||
				zoneIndex < 0 ||
				zoneIndex >= zones.length ||
				seen.has(zoneIndex)
			) {
				continue;
			}
			const zone = zones[zoneIndex];
			if (!zone) {
				continue;
			}
			seen.add(zoneIndex);
			sections.push(buildZoneTooltipSection(zone));
		}
		if (sections.length === 0) {
			return '';
		}
		const separator = '<hr class="my-2 border-0 border-t border-slate-200" />';
		return `<div class="max-h-80 overflow-y-auto space-y-3">${sections.join(separator)}</div>`;
	};

	const getOrCreatePopup = (targetMap: maplibregl.Map) => {
		let popup = mapPopups.get(targetMap);
		if (!popup) {
			popup = new maplibregl.Popup({
				closeButton: false,
				closeOnClick: false,
				offset: 12,
				className: 'rentals-tooltip'
			});
			mapPopups.set(targetMap, popup);
		}
		popup.setMaxWidth('600px');
		return popup;
	};

	const showPopup = (targetMap: maplibregl.Map, lngLat: maplibregl.LngLatLike, html: string) => {
		if (!html) {
			hidePopup(targetMap);
			return;
		}
		const popup = getOrCreatePopup(targetMap);
		popup.setLngLat(lngLat).setHTML(html).addTo(targetMap);
	};

	const hidePopup = (targetMap: maplibregl.Map) => {
		const popup = mapPopups.get(targetMap);
		popup?.remove();
	};

	const showZoneTooltip = (
		targetMap: maplibregl.Map,
		event: maplibregl.MapMouseEvent,
		featureList?: maplibregl.MapGeoJSONFeature[]
	) => {
		const features =
			featureList && featureList.length > 0
				? (featureList as maplibregl.MapGeoJSONFeature[])
				: (targetMap.queryRenderedFeatures(event.point, {
						layers: [ZONES_LAYER_ID]
					}) as maplibregl.MapGeoJSONFeature[]);
		const html = buildZoneTooltipHtml(features);
		if (html) {
			showPopup(targetMap, event.lngLat, html);
			return;
		}
		hidePopup(targetMap);
	};

	const attachStyleListener = (targetMap: maplibregl.Map) => {
		if (styleListeners.has(targetMap)) {
			return;
		}
		const handler = () => {
			ensureSourcesAndLayers(targetMap);
			setStationSourceData(stationData, targetMap);
			setVehicleSourceData(vehicleDataByIcon, targetMap);
			setZoneSourceData(zoneData, targetMap);
		};
		targetMap.on('styledata', handler);
		styleListeners.set(targetMap, handler);
	};

	const detachStyleListener = (targetMap: maplibregl.Map) => {
		const handler = styleListeners.get(targetMap);
		if (!handler) {
			return;
		}
		targetMap.off('styledata', handler);
		styleListeners.delete(targetMap);
	};

	const attachPointerEvents = (targetMap: maplibregl.Map) => {
		if (pointerHandlers.has(targetMap)) {
			return;
		}
		const stationEnter = (event: maplibregl.MapLayerMouseEvent) => {
			targetMap.getCanvas().style.cursor = 'pointer';
			const html = buildStationTooltipHtml(event.features?.[0]);
			showPopup(targetMap, event.lngLat, html);
		};
		const stationMove = (event: maplibregl.MapLayerMouseEvent) => {
			const html = buildStationTooltipHtml(event.features?.[0]);
			showPopup(targetMap, event.lngLat, html);
		};
		const stationLeave = () => {
			targetMap.getCanvas().style.cursor = '';
			hidePopup(targetMap);
		};
		const vehicleEnter = (event: maplibregl.MapLayerMouseEvent) => {
			targetMap.getCanvas().style.cursor = 'pointer';
			const html = buildVehicleTooltipHtml(event.features?.[0]);
			showPopup(targetMap, event.lngLat, html);
		};
		const vehicleMove = (event: maplibregl.MapLayerMouseEvent) => {
			const html = buildVehicleTooltipHtml(event.features?.[0]);
			showPopup(targetMap, event.lngLat, html);
		};
		const vehicleLeave = () => {
			targetMap.getCanvas().style.cursor = '';
			hidePopup(targetMap);
		};
		const vehicleClusterEnter = () => {
			targetMap.getCanvas().style.cursor = 'pointer';
			hidePopup(targetMap);
		};
		const vehicleClusterLeave = () => {
			targetMap.getCanvas().style.cursor = '';
			hidePopup(targetMap);
		};
		const zoneEnter = () => {
			targetMap.getCanvas().style.cursor = 'pointer';
		};
		const zoneLeave = () => {
			targetMap.getCanvas().style.cursor = '';
		};
		const zoneClick = (event: maplibregl.MapLayerMouseEvent) => {
			showZoneTooltip(
				targetMap,
				event,
				event.features as maplibregl.MapGeoJSONFeature[] | undefined
			);
		};
		const zoneMapClick = (event: maplibregl.MapMouseEvent) => {
			showZoneTooltip(targetMap, event);
		};
		targetMap.on('mouseenter', STATIONS_LAYER_ID, stationEnter);
		targetMap.on('mousemove', STATIONS_LAYER_ID, stationMove);
		targetMap.on('mouseleave', STATIONS_LAYER_ID, stationLeave);
		targetMap.on('mouseenter', ZONES_LAYER_ID, zoneEnter);
		targetMap.on('mouseleave', ZONES_LAYER_ID, zoneLeave);
		targetMap.on('click', ZONES_LAYER_ID, zoneClick);
		targetMap.on('click', zoneMapClick);
		for (const layerId of vehicle_point_layer_ids) {
			targetMap.on('mouseenter', layerId, vehicleEnter);
			targetMap.on('mousemove', layerId, vehicleMove);
			targetMap.on('mouseleave', layerId, vehicleLeave);
		}
		for (const layerId of vehicle_cluster_layer_ids) {
			targetMap.on('mouseenter', layerId, vehicleClusterEnter);
			targetMap.on('mouseleave', layerId, vehicleClusterLeave);
		}
		pointerHandlers.set(targetMap, {
			stationEnter,
			stationMove,
			stationLeave,
			vehicleEnter,
			vehicleMove,
			vehicleLeave,
			vehicleClusterEnter,
			vehicleClusterLeave,
			vehicleLayerIds: vehicle_point_layer_ids.slice(),
			vehicleClusterLayerIds: vehicle_cluster_layer_ids.slice(),
			zoneEnter,
			zoneLeave,
			zoneClick,
			zoneMapClick,
			zoneLayerId: ZONES_LAYER_ID
		});
	};

	const detachPointerEvents = (targetMap: maplibregl.Map) => {
		const handlers = pointerHandlers.get(targetMap);
		if (!handlers) {
			return;
		}
		targetMap.off('mouseenter', STATIONS_LAYER_ID, handlers.stationEnter);
		targetMap.off('mousemove', STATIONS_LAYER_ID, handlers.stationMove);
		targetMap.off('mouseleave', STATIONS_LAYER_ID, handlers.stationLeave);
		for (const layerId of handlers.vehicleLayerIds) {
			targetMap.off('mouseenter', layerId, handlers.vehicleEnter);
			targetMap.off('mousemove', layerId, handlers.vehicleMove);
			targetMap.off('mouseleave', layerId, handlers.vehicleLeave);
		}
		for (const layerId of handlers.vehicleClusterLayerIds) {
			targetMap.off('mouseenter', layerId, handlers.vehicleClusterEnter);
			targetMap.off('mouseleave', layerId, handlers.vehicleClusterLeave);
		}
		targetMap.off('mouseenter', handlers.zoneLayerId, handlers.zoneEnter);
		targetMap.off('mouseleave', handlers.zoneLayerId, handlers.zoneLeave);
		targetMap.off('click', handlers.zoneLayerId, handlers.zoneClick);
		targetMap.off('click', handlers.zoneMapClick);
		pointerHandlers.delete(targetMap);
		hidePopup(targetMap);
	};

	const attachClusterClickHandler = (targetMap: maplibregl.Map) => {
		let handlers = clusterClickHandlers.get(targetMap);
		if (!handlers) {
			handlers = new Map();
			clusterClickHandlers.set(targetMap, handlers);
		}
		for (const config of vehicle_source_configs) {
			if (handlers.has(config.cluster_layer_id)) {
				continue;
			}
			const handler = async (event: maplibregl.MapLayerMouseEvent) => {
				const feature = event.features?.[0];
				if (!feature) {
					return;
				}
				const clusterId = feature.properties?.cluster_id;
				if (typeof clusterId !== 'number') {
					return;
				}
				const source = targetMap.getSource(config.source_id) as
					| maplibregl.GeoJSONSource
					| undefined;
				if (!source || typeof source.getClusterExpansionZoom !== 'function') {
					return;
				}
				try {
					const zoomLevel = await source.getClusterExpansionZoom(clusterId);
					if (typeof zoomLevel !== 'number') {
						return;
					}
					if (feature.geometry?.type !== 'Point') {
						return;
					}
					const coordinates = feature.geometry.coordinates;
					if (!Array.isArray(coordinates) || coordinates.length < 2) {
						return;
					}
					const [lng, lat] = coordinates as [number, number];
					const maxZoom =
						typeof targetMap.getMaxZoom === 'function' ? targetMap.getMaxZoom() : undefined;
					const targetZoom =
						maxZoom !== undefined ? Math.min(zoomLevel + 0.5, maxZoom) : zoomLevel + 0.5;
					targetMap.easeTo({
						center: [lng, lat],
						zoom: targetZoom
					});
				} catch (err) {
					console.error('Failed to get cluster expansion zoom', err);
					return;
				}
			};
			targetMap.on('click', config.cluster_layer_id, handler);
			handlers.set(config.cluster_layer_id, handler);
		}
	};

	const detachClusterClickHandler = (targetMap: maplibregl.Map) => {
		const handlers = clusterClickHandlers.get(targetMap);
		if (!handlers) {
			return;
		}
		for (const [layerId, handler] of handlers) {
			targetMap.off('click', layerId, handler);
		}
		clusterClickHandlers.delete(targetMap);
	};

	const initMapIntegration = (targetMap: maplibregl.Map) => {
		ensureSourcesAndLayers(targetMap);
		setStationSourceData(stationData, targetMap);
		setVehicleSourceData(vehicleDataByIcon, targetMap);
		setZoneSourceData(zoneData, targetMap);
		attachStyleListener(targetMap);
		attachPointerEvents(targetMap);
		attachClusterClickHandler(targetMap);
	};

	const cleanupMapIntegration = (targetMap: maplibregl.Map) => {
		detachPointerEvents(targetMap);
		detachStyleListener(targetMap);
		detachClusterClickHandler(targetMap);
		hidePopup(targetMap);
		mapPopups.delete(targetMap);
		removeSourcesAndLayers(targetMap);
	};

	const emptyStationCollection = () => ({
		type: 'FeatureCollection' as const,
		features: [] as Feature<Point, StationFeatureProperties>[]
	});

	const resetCachedData = () => {
		stationById = new Map<string, RentalStation>();
		vehicleById = new Map<string, RentalVehicle>();
		providerById = new Map<string, RentalProvider>();
		zones = [];
	};

	const setEmptyCollections = (targetMap: maplibregl.Map | undefined) => {
		resetCachedData();
		setStationSourceData(emptyStationCollection(), targetMap);
		setVehicleSourceData(create_empty_vehicle_collections(), targetMap);
		setZoneSourceData(create_empty_zone_collection(), targetMap);
	};

	const fetchRentals = async () => {
		if (!map || !bounds || zoom <= MIN_ZOOM) {
			loadedBounds = null;
			return;
		}

		const mapBounds = maplibregl.LngLatBounds.convert(bounds);

		if (containsBounds(loadedBounds, mapBounds)) {
			return;
		}

		const expandedBounds = expandBounds(mapBounds);
		const max = lngLatToStr(expandedBounds.getNorthWest());
		const min = lngLatToStr(expandedBounds.getSouthEast());
		const token = ++requestToken;

		try {
			const { data, error } = await rentals({
				query: { max, min, withZones: showZones }
			});

			if (token !== requestToken) {
				return;
			}

			if (error) {
				console.error('Failed to load rental data', error);
				loadedBounds = null;
				return;
			}

			const providers = data?.providers ?? [];
			providerById = new Map<string, RentalProvider>(providers.map((p) => [p.id, p]));
			const stations = data?.stations ?? [];
			stationById = new Map<string, RentalStation>(
				stations.map((station) => [createScopedId(station.providerId, station.id), station])
			);
			const vehicles = data?.vehicles ?? [];
			vehicleById = new Map<string, RentalVehicle>(
				vehicles.map((vehicle) => [createScopedId(vehicle.providerId, vehicle.id), vehicle])
			);
			zones = data?.zones ?? [];
			setZoneSourceData(buildZoneFeatures(zones), map);
			setStationSourceData(buildStationFeatures(stations), map);
			setVehicleSourceData(buildVehicleFeatures(vehicles), map);
			loadedBounds = expandedBounds;
		} catch (error) {
			console.error('Failed to load rental data', error);
			loadedBounds = null;
		}
	};

	$effect(() => {
		if (map === activeMap) {
			return;
		}
		if (activeMap) {
			cleanupMapIntegration(activeMap);
		}
		activeMap = map;
		if (map) {
			initMapIntegration(map);
		}
	});

	$effect(() => {
		if (!map || !bounds) {
			loadedBounds = null;
			setEmptyCollections(map);
			return;
		}

		if (zoom <= MIN_ZOOM) {
			loadedBounds = null;
			setEmptyCollections(map);
			return;
		}

		fetchRentals();
	});

	onDestroy(() => {
		if (activeMap) {
			cleanupMapIntegration(activeMap);
		}
	});
</script>
