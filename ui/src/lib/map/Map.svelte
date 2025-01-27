<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { setContext, type Snippet } from 'svelte';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { createShield } from './shield';

	let {
		map = $bindable(),
		zoom = $bindable(),
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		bounds = $bindable(),
		style,
		attribution,
		transformRequest,
		center,
		children,
		class: className
	}: {
		map?: maplibregl.Map;
		style: maplibregl.StyleSpecification | undefined;
		attribution: string;
		transformRequest?: maplibregl.RequestTransformFunction;
		center: maplibregl.LngLatLike;
		bounds?: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		children?: Snippet;
		class: string;
	} = $props();

	let currStyle: maplibregl.StyleSpecification | undefined = undefined;
	let ctx = $state<{ map: maplibregl.Map | undefined }>({ map: undefined });
	setContext('map', ctx);

	let currentZoom: number | undefined = undefined;
	let currentCenter: maplibregl.LngLatLike | undefined = undefined;

	const createMap = (container: HTMLElement) => {
		let tmp = new maplibregl.Map({
			container,
			zoom,
			bounds,
			center,
			style,
			transformRequest,
			attributionControl: { customAttribution: attribution }
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

		tmp.on('load', () => {
			tmp.setZoom(zoom);
			tmp.setCenter(center);
			currentZoom = zoom;
			currentCenter = center;
			bounds = tmp.getBounds();
			map = tmp;
			ctx.map = tmp;
			currStyle = style;
			currentZoom = zoom;
		});

		tmp.on('moveend', async () => {
			zoom = tmp.getZoom();
			currentZoom = zoom;
			bounds = tmp.getBounds();
		});

		return {
			destroy() {
				tmp.remove();
				ctx.map = undefined;
			}
		};
	};

	$effect(() => {
		if (style != currStyle && ctx.map) {
			ctx.map.setStyle(style || null);
		}
	});

	$effect(() => {
		if (map && $state.snapshot(zoom) !== currentZoom) {
			map.setZoom(zoom);
			currentZoom = zoom;
		}
	});

	$effect(() => {
		if (map && center != currentCenter) {
			map.setCenter(center);
			currentCenter = center;
		}
	});
</script>

<div use:createMap class={className}>
	{#if children}
		{@render children()}
	{/if}
</div>
