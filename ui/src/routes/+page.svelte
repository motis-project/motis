<script lang="ts">
	import maplibregl from 'maplibre-gl';
	import {
		MapLibre,
		Control,
		ControlGroup,
		ControlButton,
		GeoJSON,
		LineLayer,
		CircleLayer
	} from 'svelte-maplibre';
	import { getStyle } from '$lib/style';
	import { createShield } from '$lib/shield';

	const baseUrl = 'https://osr.motis-project.de';

	let zoom = $state(18);
	let bounds = $state<undefined | maplibregl.LngLatBoundsLike>(undefined);

	const getGraph = async (bounds: maplibregl.LngLatBounds) => {
		const response = await fetch(`${baseUrl}/api/graph`, {
			method: 'POST',
			mode: 'cors',
			headers: {
				'Access-Control-Allow-Origin': '*',
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				level: level,
				waypoints: bounds.toArray().flat()
			})
		});
		return await response.json();
	};

	const getLevels = async (bounds: maplibregl.LngLatBounds) => {
		const response = await fetch(`${baseUrl}/api/levels`, {
			method: 'POST',
			mode: 'cors',
			headers: {
				'Access-Control-Allow-Origin': '*',
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				waypoints: bounds.toArray().flat()
			})
		});
		return await response.json();
	};

	let level = $state(0);
	let levels = $state<null | Array<number>>(null);
	$effect(async () => {
		levels =
			bounds === undefined || zoom < 18
				? null
				: await getLevels(maplibregl.LngLatBounds.convert(bounds!));
	});

	let map = $state<maplibregl.Map | null>(null);
	let showGraph = $state(false);
	let graph = $state<null | Object>(null);
	$effect(async () => {
		bounds; // trigger graph reload when bounds change
		graph = showGraph && map ? await getGraph(map.getBounds()) : null;
	});

	const [shieldData, shieldOptions] = createShield({
		fill: 'hsl(0, 0%, 98%)',
		stroke: 'hsl(0, 0%, 75%)'
	});
</script>

<MapLibre
	bind:map
	bind:bounds
	bind:zoom
	standardControls
	transformRequest={(url, _resourceType) => {
		if (url.startsWith('/')) {
			return { url: `https://europe.motis-project.de/tiles${url}` };
		}
	}}
	center={[8.663351200419433, 50.10680913598618]}
	class="h-screen"
	images={shieldData == null || shieldOptions == null
		? []
		: [{ id: 'shield', data: shieldData, options: shieldOptions }]}
	style={getStyle(level)}
>
	<Control class="flex flex-col gap-y-2">
		<ControlGroup>
			<ControlButton
				class={showGraph ? 'bg-green-200' : ''}
				on:click={() => {
					showGraph = !showGraph;
				}}>G</ControlButton
			>
		</ControlGroup>

		<ControlGroup>
			{#if levels}
				{#each levels as l}
					<ControlButton
						on:click={() => {
							level = l;
						}}
					>
						{l}
					</ControlButton>
				{/each}
			{/if}
		</ControlGroup>
	</Control>

	{#if graph != null}
		<GeoJSON id="graph" data={graph} promoteId="GRAPH">
			<LineLayer
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
			<LineLayer
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
			<CircleLayer
				filter={['all', ['==', '$type', 'Point']]}
				paint={{
					'circle-color': ['match', ['get', 'label'], 'unreachable', '#ff1150', '#11ffaf'],
					'circle-radius': 6
				}}
			/>
		</GeoJSON>
	{/if}
</MapLibre>
