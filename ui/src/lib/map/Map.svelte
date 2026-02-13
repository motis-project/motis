<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { setContext, type Snippet } from 'svelte';
	import 'maplibre-gl/dist/maplibre-gl.css';
	import { createShield } from './shield';
	import { browser } from '$app/environment';
	let {
		map = $bindable(),
		zoom = $bindable(),
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
