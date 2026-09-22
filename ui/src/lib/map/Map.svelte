<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { setContext, type Snippet } from 'svelte';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { createShield } from './shield';
	import { browser } from '$app/environment';
	// pinned to 0.2.3 — 0.4.0's `exports` field blocks deep-importing the worker script
	import rtlTextUrl from '@mapbox/mapbox-gl-rtl-text/mapbox-gl-rtl-text.min.js?url';

	// required for correct rendering of RTL scripts (Arabic, Hebrew, ...);
	// lazy: only loaded once RTL text is actually encountered
	if (browser && maplibregl.getRTLTextPluginStatus() === 'unavailable') {
		maplibregl.setRTLTextPlugin(rtlTextUrl, true);
	}
	let {
		map = $bindable(),
		zoom = $bindable(),
		bounds = $bindable(),
		center = $bindable(),
		bearing = $bindable(),
		style,
		attribution,
		transformRequest,
		children,
		class: className
	}: {
		map?: maplibregl.Map;
		style: maplibregl.StyleSpecification | undefined;
		attribution: string | undefined | false;
		transformRequest?: maplibregl.RequestTransformFunction;
		center: maplibregl.LngLatLike;
		bounds?: maplibregl.LngLatBoundsLike | undefined;
		bearing?: number | undefined;
		zoom: number;
		children?: Snippet;
		class: string;
	} = $props();

	let el: HTMLElement | null = null;
	let currStyle: maplibregl.StyleSpecification | undefined = style;
	let ctx = $state<{ map: maplibregl.Map | undefined }>({ map: undefined });
	let touchStartTime = $state<number | null>(null);
	let touchLocation = $state<{ x: number; y: number } | null>(null);
	setContext('map', ctx);

	const updateStyle = () => {
		if (style != currStyle) {
			if (!ctx.map && el) {
				createMap(el);
			} else if (ctx.map) {
				ctx.map.setStyle(style || null);
			}
			currStyle = style;
		}
	};
	const createMap = (container: HTMLElement) => {
		if (!style) {
			return;
		}
		let tmp: maplibregl.Map;
		try {
			tmp = new maplibregl.Map({
				hash: true,
				container,
				zoom,
				bounds,
				center,
				style,
				pitchWithRotate: false,
				fadeDuration: 0,
				transformRequest,
				attributionControl:
					attribution === false || attribution === undefined
						? attribution
						: { customAttribution: attribution }
			});
			tmp.addImage(
				'shield',
				...createShield({
					fill: 'hsl(0, 0%, 98%)',
					stroke: 'hsl(0, 0%, 75%)'
				})
			);

			tmp.addImage(
				'shield-dark',
				...createShield({
					fill: 'hsl(0, 0%, 16%)',
					stroke: 'hsl(0, 0%, 30%)'
				})
			);

			const scale = new maplibregl.ScaleControl({
				maxWidth: 100,
				unit: 'metric'
			});

			tmp.addControl(scale, browser && window.innerWidth < 768 ? 'top-left' : 'bottom-left');

			tmp.on('load', () => {
				map = tmp;
				ctx.map = tmp;
				bounds = tmp.getBounds();
				tmp.on('moveend', () => {
					zoom = tmp.getZoom();
					center = tmp.getCenter();
					bounds = tmp.getBounds();
				});
				tmp.on('rotate', () => {
					bearing = tmp.getBearing();
				});
				tmp.on('touchstart', (event) => {
					touchStartTime = new Date().getTime();
					touchLocation = { x: event.point.x, y: event.point.y };
				});
				tmp.on('touchend', (event) => {
					const longTouchTimeMS = 500;
					const acceptableMoveDistance = 20;

					if (touchStartTime && touchLocation) {
						const touchTime = new Date().getTime() - touchStartTime;
						const didNotMoveMap =
							Math.abs(event.point.x - touchLocation.x) < acceptableMoveDistance &&
							Math.abs(event.point.y - touchLocation.y) < acceptableMoveDistance;

						if (touchTime > longTouchTimeMS && didNotMoveMap) {
							tmp.fire('contextmenu', { lngLat: event.lngLat });
						}
					}

					touchStartTime = null;
					touchLocation = null;
				});
			});
		} catch (e) {
			console.log(e);
		}

		return {
			destroy() {
				tmp?.remove();
				ctx.map = undefined;
			}
		};
	};

	$effect(updateStyle);
</script>

<div use:createMap bind:this={el} class={className}>
	{#if children}
		{@render children()}
	{/if}
</div>
