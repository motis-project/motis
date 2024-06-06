<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import { getStyle } from '$lib/style';
	import Map from '$lib/Map.svelte';
	import Control from '$lib/Control.svelte';
	import GeoJSON from '$lib/GeoJSON.svelte';
	import Layer from '$lib/Layer.svelte';
	import { getGraph, getLevels, getMatches } from '$lib/api';
	import { toTable } from '$lib/toTable';

	let zoom = $state(18);
	let bounds = $state<undefined | maplibregl.LngLatBounds>(undefined);
	let map = $state<null | maplibregl.Map>(null);

	let level = $state(0);
	let levels = $derived.by(async () =>
		bounds === undefined || zoom < 18
			? null
			: await getLevels(maplibregl.LngLatBounds.convert(bounds))
	);

	let showGraph = $state(false);
	let graph = $state<null | Object>(null);
	$effect(async () => {
		graph = showGraph && bounds ? await getGraph(bounds, level) : null;
	});

	let showMatches = $state(false);
	let matches = $state<null | Object>(null);
	$effect(async () => {
		matches = showMatches && bounds ? await getMatches(bounds) : null;
	});

	let init = false;
	$effect(() => {
		if (map && !init) {
			['graph-node', 'graph-edge', 'graph-geometry', 'matches', 'match'].forEach((layer) => {
				map!.on('click', layer, (e) => {
					new maplibregl.Popup()
						.setLngLat(e.lngLat)
						.setDOMContent(toTable(e.features[0].properties))
						.addTo(map!);
					e.originalEvent.stopPropagation();
				});

				map!.on('mouseenter', layer, () => {
					map!.getCanvas().style.cursor = 'pointer';
				});

				map!.on('mouseleave', layer, () => {
					map!.getCanvas().style.cursor = '';
				});
			});
			init = true;
		}
	});
</script>

<Map
	bind:map
	bind:bounds
	transformRequest={(url, _resourceType) => {
		if (url.startsWith('/')) {
			return { url: `https://europe.motis-project.de/tiles${url}` };
		}
	}}
	center={[8.663351200419433, 50.10680913598618]}
	zoom={18}
	class="h-screen"
	style={getStyle(level)}
>
	<Control
		class={showGraph ? '!bg-green-200' : ''}
		onclick={() => {
			showGraph = !showGraph;
		}}>G</Control
	>
	<Control
		class={showMatches ? '!bg-green-200' : ''}
		onclick={() => {
			showMatches = !showMatches;
		}}>M</Control
	>

	{#if levels}
		{#await levels then lvls}
			{#each lvls as l}
				<Control
					onclick={() => {
						level = l;
					}}
				>
					{l}
				</Control>
			{/each}
		{/await}
	{/if}

	{#if graph != null}
		<GeoJSON id="graph" data={graph}>
			<Layer
				id="graph-geometry"
				type="line"
				filter={[
					'all',
					['==', 'type', 'geometry'],
					['any', ['!has', 'level'], ['==', 'level', level]]
				]}
				layout={{
					'line-join': 'round',
					'line-cap': 'round'
				}}
				paint={{
					'line-color': '#e55e5e',
					'line-width': 3,
					'line-opacity': 1
				}}
			/>
			<Layer
				id="graph-edges"
				type="line"
				filter={['all', ['==', 'type', 'edge'], ['any', ['!has', 'level'], ['==', 'level', level]]]}
				layout={{
					'line-join': 'round',
					'line-cap': 'round'
				}}
				paint={{
					'line-color': '#a300d9',
					'line-width': 3
				}}
			/>
			<Layer
				id="graph-nodes"
				type="circle"
				filter={['all', ['==', '$type', 'Point']]}
				layout={{}}
				paint={{
					'circle-color': ['match', ['get', 'label'], 'unreachable', '#ff1150', '#11ffaf'],
					'circle-radius': 6
				}}
			/>
		</GeoJSON>
	{/if}

	{#if matches != null}
		<GeoJSON id="matches" data={matches}>
			<Layer
				id="matches"
				type="circle"
				filter={['all', ['==', '$type', 'Point']]}
				layout={{}}
				paint={{
					'circle-color': ['match', ['get', 'type'], 'location', '#ff0000', '#0000ff'],
					'circle-radius': 5
				}}
			/>
			<Layer
				id="match"
				type="line"
				filter={['all', ['==', 'type', 'match']]}
				layout={{
					'line-join': 'round',
					'line-cap': 'round'
				}}
				paint={{
					'line-color': '#00ff00',
					'line-width': 3
				}}
			/>
		</GeoJSON>
	{/if}
</Map>
