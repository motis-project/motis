<script lang="ts">
	import type { Map } from 'maplibre-gl';
	import type { PrePostDirectMode } from '$lib/Modes';
	import { IsochronesLayer, type IsochronesCircle } from '$lib/map/IsochronesLayer';
	import type { IsochronesOptions, IsochronesPos } from '$lib/map/IsochronesShared';

	let {
		map,
		isochronesData,
		streetModes,
		wheelchair,
		maxAllTime,
		maxTravelTime,
		active,
		options
	}: {
		map: Map | undefined;
		isochronesData: IsochronesPos[];
		streetModes: PrePostDirectMode[];
		wheelchair: boolean;
		maxAllTime: number;
		maxTravelTime: number;
		active: boolean;
		options: IsochronesOptions;
	} = $props();

	const LAYER_ID = 'isochrones';

	const metersPerSecond = $derived(
		// Should match the speed used for routing
		streetModes.includes('BIKE') ? 3.8 : wheelchair ? 0.8 : 1.2
	);

	const circles = $derived(
		isochronesData.map((pos: IsochronesPos): IsochronesCircle => {
			// Walking outwards spends the remaining budget one second at a time,
			// but never more than the pre/post transit limit allows.
			const centerSeconds = Math.max(0, pos.seconds);
			const walkSeconds = Math.min(centerSeconds, maxAllTime);
			return {
				lng: pos.lng,
				lat: pos.lat,
				radiusMeters: walkSeconds * metersPerSecond,
				centerSeconds,
				edgeSeconds: centerSeconds - walkSeconds
			};
		})
	);

	let layer = $state<IsochronesLayer>();

	$effect(() => {
		if (!map) {
			return;
		}
		const m = map;
		const isochrones = new IsochronesLayer(LAYER_ID);
		layer = isochrones;

		// The style is replaced whenever the theme, the level or the hillshades
		// change, which drops all layers. Add the layer back once it is gone.
		const addLayer = () => {
			if (m.getLayer(LAYER_ID) || !m.isStyleLoaded()) {
				return;
			}
			try {
				m.addLayer(isochrones);
			} catch (e) {
				console.error('[Isochrones] failed to add layer', e);
			}
		};

		addLayer();
		m.on('styledata', addLayer);
		m.on('idle', addLayer);

		return () => {
			m.off('styledata', addLayer);
			m.off('idle', addLayer);
			if (m.getLayer(LAYER_ID)) {
				m.removeLayer(LAYER_ID);
			}
			isochrones.cleanup();
			layer = undefined;
		};
	});

	$effect(() => {
		layer?.setCircles(circles, maxTravelTime);
	});

	$effect(() => {
		layer?.setOpacity(options.opacity / 1000);
	});

	$effect(() => {
		layer?.setActive(active);
	});
</script>
