<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { setContext, untrack, type Snippet } from 'svelte';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { createShield } from './shield';

	let {
		map = $bindable(),
		zoom = $bindable(),
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		bounds = $bindable(),
		center = $bindable(),
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
		zoom: number;
		children?: Snippet;
		class: string;
	} = $props();

	let el: HTMLElement | null = null;
	let currStyle: maplibregl.StyleSpecification | undefined = style;
	let ctx = $state<{ map: maplibregl.Map | undefined }>({ map: undefined });
	setContext('map', ctx);

	let currentZoom: number | undefined = undefined;
	let currentCenter: maplibregl.LngLatLike | undefined = undefined;

	const createMap = (container: HTMLElement) => {
		if (!style) {
			return;
		}
		let tmp: maplibregl.Map;
		try {
			tmp = new maplibregl.Map({
				container,
				zoom,
				bounds,
				center,
				style,
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

			tmp.on('load', () => {
				tmp.setZoom(zoom);
				tmp.setCenter(center);
				currentZoom = zoom;
				currentCenter = center;
				bounds = tmp.getBounds();
				currStyle = style;
				map = tmp;
				ctx.map = tmp;
				currentZoom = zoom;
			});

			tmp.on('moveend', () =>
				untrack(async () => {
					zoom = tmp.getZoom();
					currentZoom = zoom;
					bounds = tmp.getBounds();
					center = tmp.getCenter();
				})
			);
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

	$effect(() => {
		if (style != currStyle) {
			if (!ctx.map && el) {
				createMap(el);
			} else if (ctx.map) {
				ctx.map.setStyle(style || null);
			}
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

<div use:createMap bind:this={el} class={className}>
	{#if children}
		{@render children()}
	{/if}
</div>
