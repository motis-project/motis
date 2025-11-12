<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import type { PointLike } from 'maplibre-gl';
	import { getContext, onDestroy } from 'svelte';

	import { ZoneFillLayer } from './zone-fill-layer';
	import type { RentalZoneFeature } from './zone-types';

	class Props {
		id!: string;
		features!: RentalZoneFeature[];
		beforeLayerId?: string;
		opacity?: number;
	}

	const ZONE_FILL_OPACITY = 0.4;

	let { id, features, beforeLayerId, opacity = ZONE_FILL_OPACITY }: Props = $props();

	type MapContext = { map: maplibregl.Map | null };
	const ctx: MapContext = getContext('map');

	let layerInstance = $state<ZoneFillLayer | null>(null);
	let pendingRetryHandler: ((event?: unknown) => void) | null = null;
	let currentFeatures: RentalZoneFeature[] | null = null;
	let lastOpacity = opacity;

	const ensureLayerInstance = () => {
		if (!layerInstance) {
			layerInstance = new ZoneFillLayer({ id, opacity });
		}
		return layerInstance;
	};

	const disposeLayerInstance = () => {
		if (!layerInstance) {
			return;
		}
		layerInstance.cleanup();
		layerInstance = null;
		currentFeatures = null;
	};

	const getLayerIndex = (mapInstance: maplibregl.Map, layerId: string) => {
		const style = mapInstance.getStyle();
		const layers = style?.layers ?? [];
		return layers.findIndex((layer) => layer.id === layerId);
	};

	const clearPendingRetry = (map: maplibregl.Map | null | undefined) => {
		if (!map || !pendingRetryHandler) {
			return;
		}
		map.off('styledata', pendingRetryHandler);
		map.off('idle', pendingRetryHandler);
		pendingRetryHandler = null;
	};

	const removeFromMap = (map: maplibregl.Map | null | undefined) => {
		const instance = layerInstance;
		if (!instance) {
			return;
		}
		if (map && map.getLayer(id)) {
			try {
				map.removeLayer(id);
			} catch (error) {
				console.error('[ZoneLayer] failed to remove layer', error);
			}
		}
		instance.setFeatures([]);
		disposeLayerInstance();
	};

	$effect(() => {
		const map = ctx.map;
		if (!map || features.length === 0) {
			clearPendingRetry(map);
			removeFromMap(map);
			currentFeatures = null;
			return;
		}

		const layer = ensureLayerInstance();
		if (lastOpacity !== opacity) {
			layer.setOpacity(opacity);
			lastOpacity = opacity;
		}
		if (features !== currentFeatures) {
			currentFeatures = features;
			layer.setFeatures(features);
		}

		const ensureLayerOrder = () => {
			if (!beforeLayerId || beforeLayerId === id) {
				return;
			}
			if (!map.getLayer(id) || !map.getLayer(beforeLayerId)) {
				return;
			}
			const layerIndex = getLayerIndex(map, id);
			const targetIndex = getLayerIndex(map, beforeLayerId);
			if (layerIndex === -1 || targetIndex === -1) {
				return;
			}
			if (layerIndex >= targetIndex) {
				map.moveLayer(id, beforeLayerId);
			}
		};

		const addLayerIfNeeded = () => {
			if (!map.getLayer(id)) {
				try {
					const before = beforeLayerId && map.getLayer(beforeLayerId) ? beforeLayerId : undefined;
					if (before) {
						map.addLayer(layer, before);
					} else {
						map.addLayer(layer);
					}
				} catch (error) {
					console.error('[ZoneLayer] failed to add layer', error);
				}
			}
			ensureLayerOrder();
		};

		const attemptAddLayer = () => {
			if (!map.isStyleLoaded()) {
				if (!pendingRetryHandler) {
					const retry = () => {
						if (!pendingRetryHandler) {
							return;
						}
						map.off('styledata', retry);
						map.off('idle', retry);
						pendingRetryHandler = null;
						attemptAddLayer();
					};
					pendingRetryHandler = retry;
					map.on('styledata', retry);
					map.on('idle', retry);
				}
				return;
			}
			addLayerIfNeeded();
		};

		const handleStyleData = () => {
			attemptAddLayer();
			ensureLayerOrder();
		};

		attemptAddLayer();
		map.on('styledata', handleStyleData);

		return () => {
			map.off('styledata', handleStyleData);
			clearPendingRetry(map);
		};
	});

	onDestroy(() => {
		const mapInstance = ctx.map;
		clearPendingRetry(mapInstance);
		removeFromMap(mapInstance);
		disposeLayerInstance();
		lastOpacity = opacity;
	});

	export function pick(point: PointLike) {
		return layerInstance?.pickFeatureAt(point) ?? null;
	}
</script>
