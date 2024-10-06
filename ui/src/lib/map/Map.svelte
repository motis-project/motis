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
		transformRequest,
		center,
		children,
		class: className
	}: {
		map?: maplibregl.Map;
		style: maplibregl.StyleSpecification;
		transformRequest?: maplibregl.RequestTransformFunction;
		center: maplibregl.LngLatLike;
		bounds?: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		children?: Snippet;
		class: string;
	} = $props();

	let currStyle: maplibregl.StyleSpecification | null = null;
	let ctx = $state<{ map: maplibregl.Map | undefined }>({ map: undefined });
	setContext('map', ctx);

	$effect(() => {
		if (style != currStyle && ctx.map) {
			ctx.map.setStyle(style);
		}
	});

	let currentCenter = center;
	const createMap = (container: HTMLElement) => {
		map = new maplibregl.Map({ container, zoom, center, style, transformRequest });

		map.addImage(
			'shield',
			...createShield({
				fill: 'hsl(0, 0%, 98%)',
				stroke: 'hsl(0, 0%, 75%)'
			})
		);

		map.addImage(
			'shield-dark',
			...createShield({
				fill: 'hsl(0, 0%, 16%)',
				stroke: 'hsl(0, 0%, 30%)'
			})
		);

		map.on('load', () => {
			currStyle = style;
			ctx.map = map;
			bounds = map?.getBounds();
		});

		map.on('moveend', async () => {
			bounds = ctx.map?.getBounds() as maplibregl.LngLatBoundsLike;
			zoom = ctx.map?.getZoom() as number;
		});

		return {
			destroy() {
				ctx.map?.remove();
				ctx.map = undefined;
			}
		};
	};

	$effect(() => {
		if (center != currentCenter) {
			map?.setCenter(center);
			currentCenter = center;
		}
	});
</script>

<div use:createMap class={className}>
	{#if children}
		{@render children()}
	{/if}
</div>
