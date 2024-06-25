<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { setContext, type Snippet } from 'svelte';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { createShield } from './shield';

	let {
		map = $bindable(),
		zoom = $bindable(),
		bounds = $bindable(),
		style,
		transformRequest,
		center,
		children,
		...props
	}: {
		map: maplibregl.Map | null;
		style: maplibregl.StyleSpecification;
		transformRequest: maplibregl.RequestTransformFunction;
		center: maplibregl.LngLatLike;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		children: Snippet;
	} = $props();

	let currStyle: maplibregl.StyleSpecification | null = null;
	let ctx = $state<{ map: maplibregl.Map | null }>({ map: null });
	setContext('map', ctx);

	$effect(() => {
		if (style != currStyle && ctx.map) {
			ctx.map.setStyle(style);
		}
	});

	const createMap = (container: HTMLElement) => {
		map = new maplibregl.Map({ container, zoom, center, style, transformRequest });

		map.addImage(
			'shield',
			...createShield({
				fill: 'hsl(0, 0%, 98%)',
				stroke: 'hsl(0, 0%, 75%)'
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
				ctx.map = null;
			}
		};
	};
</script>

<div use:createMap {...props}>
	{@render children()}
</div>
