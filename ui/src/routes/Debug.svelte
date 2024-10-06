<script lang="ts">
	import Bug from 'lucide-svelte/icons/bug';
	import { Button } from '$lib/components/ui/button';
	import GeoJSON from '$lib/map/GeoJSON.svelte';
	import Layer from '$lib/map/Layer.svelte';
	import {
		Table,
		TableBody,
		TableCell,
		TableHead,
		TableHeader,
		TableRow
	} from '$lib/components/ui/table';
	import maplibregl from 'maplibre-gl';
	import { footpaths } from '$lib/openapi';
	import Control from '$lib/map/Control.svelte';
	import { Card } from '$lib/components/ui/card';
	import Marker from '$lib/map/Marker.svelte';
	import { posToLocation, type Location as ApiLocation } from '$lib/Location';
	import geojson from 'geojson';
	import Popup from '$lib/map/Popup.svelte';

	const baseUrl = 'http://localhost:8080';

	const post = async (path: string, req: unknown) => {
		const response = await fetch(`${baseUrl}${path}`, {
			method: 'POST',
			mode: 'cors',
			headers: {
				'Access-Control-Allow-Origin': '*',
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(req)
		});
		return await response.json();
	};

	type Location = {
		lat: number;
		lng: number;
		level: number;
	};

	type RoutingQuery = {
		start: Location;
		destination: Location;
		profile: string;
		direction: string;
	};

	const toLocation = (l: ApiLocation): Location => {
		return {
			lat: l.value.match!.lat,
			lng: l.value.match!.lon,
			level: l.value.match!.level
		};
	};

	const getRoute = async (query: RoutingQuery) => {
		return await post('/api/route', query);
	};

	const getMatches = async (bounds: maplibregl.LngLatBounds) => {
		return await post('/api/matches', bounds.toArray().flat());
	};

	export const getGraph = async (bounds: maplibregl.LngLatBounds, level: number) => {
		return await post('/api/graph', {
			level: level,
			waypoints: bounds.toArray().flat()
		});
	};

	let {
		bounds,
		level
	}: {
		bounds: maplibregl.LngLatBoundsLike | undefined;
		level: number;
	} = $props();

	let debug = $state(false);
	let id = $state<string>();
	let fps = $derived(
		id && bounds && debug ? footpaths<false>({ query: { id } }).then((x) => x.data) : undefined
	);
	let matches = $derived(
		bounds && debug ? getMatches(maplibregl.LngLatBounds.convert(bounds)) : undefined
	);

	let graph = $state<null | geojson.GeoJSON>(null);
	$effect(() => {
		if (debug && bounds) {
			getGraph(maplibregl.LngLatBounds.convert(bounds), level).then((response: geojson.GeoJSON) => {
				graph = response;
			});
		} else {
			graph = null;
		}
	});

	let profile = $state<string>();
	let start = $state.raw<ApiLocation>();
	let destination = $state.raw<ApiLocation>();
	let route = $derived(
		start?.value.match?.lat &&
			start?.value.match.lon &&
			destination?.value.match?.lat &&
			destination?.value.match.lon &&
			profile
			? getRoute({
					start: toLocation(start),
					destination: toLocation(destination),
					profile,
					direction: 'forward'
				})
			: undefined
	);
</script>

<Button
	size="icon"
	variant={debug ? 'default' : 'outline'}
	on:click={() => {
		debug = !debug;
	}}
>
	<Bug size="icon" class="h-[1.2rem] w-[1.2rem]" />
</Button>

{#if debug}
	{#if fps}
		{#await fps then f}
			{#if f}
				<Control position="bottom-right">
					<Card class="w-[500px] h-[500px] overflow-y-auto bg-background rounded-lg">
						<Table>
							<TableHeader>
								<TableRow>
									<TableHead>Station</TableHead>
									<TableHead>Default</TableHead>
									<TableHead>Foot</TableHead>
									<TableHead>Wheelchair</TableHead>
								</TableRow>
							</TableHeader>
							<TableBody>
								{#each f.footpaths as x}
									<TableRow>
										<TableCell>{x.to.name}</TableCell>
										<TableCell>
											{#if x.default !== undefined}
												<Button
													variant="outline"
													on:click={() => {
														start = posToLocation(f.place);
														destination = posToLocation(x.to);
														profile = 'foot';
													}}
												>
													{x.default}
												</Button>
											{/if}
										</TableCell>
										<TableCell>
											{#if x.foot !== undefined}
												<Button
													variant="outline"
													on:click={() => {
														start = posToLocation(f.place);
														destination = posToLocation(x.to);
														profile = 'foot';
													}}
												>
													{x.foot}
												</Button>
											{/if}
										</TableCell>
										<TableCell>
											{#if x.wheelchair !== undefined}
												<Button
													class={x.wheelchairUsesElevator ? 'text-red-500' : 'text-green-500'}
													variant="outline"
													on:click={() => {
														start = posToLocation(f.place);
														destination = posToLocation(x.to);
														profile = 'wheelchair';
													}}
												>
													{x.wheelchair}
												</Button>
											{/if}
										</TableCell>
									</TableRow>
								{/each}
							</TableBody>
						</Table>
					</Card>
				</Control>
			{/if}
		{/await}
	{/if}

	{#if matches}
		{#await matches then m}
			<GeoJSON id="matches" data={m}>
				<Layer
					onclick={(e) => {
						const props = e.features![0].properties;
						id = props.id;
					}}
					id="matches"
					type="circle"
					filter={['all', ['==', '$type', 'Point']]}
					layout={{}}
					paint={{
						'circle-color': ['match', ['get', 'type'], 'location', '#34ebde', '#fa921b'],
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
		{/await}
	{/if}

	{#if route}
		{#await route then r}
			{#if r.type == 'FeatureCollection'}
				<GeoJSON id="route" data={r}>
					<Layer
						id="path-outline"
						type="line"
						layout={{
							'line-join': 'round',
							'line-cap': 'round'
						}}
						filter={['any', ['!has', 'level'], ['==', 'level', level]]}
						paint={{
							'line-color': '#cfa900',
							'line-width': 7.5,
							'line-opacity': 0.8
						}}
					/>
					<Layer
						id="path"
						type="line"
						layout={{
							'line-join': 'round',
							'line-cap': 'round'
						}}
						filter={['any', ['!has', 'level'], ['==', 'level', level]]}
						paint={{
							'line-color': '#fccf03',
							'line-width': 5,
							'line-opacity': 0.8
						}}
					/>
				</GeoJSON>
			{/if}
		{/await}
	{/if}

	<!-- eslint-disable-next-line -->
	{#snippet nodeDetails(_1: maplibregl.MapMouseEvent, _2: () => void, features: any)}
		<Table>
			<TableBody>
				{#each Object.entries(features[0].properties) as [key, value]}
					<TableRow>
						<TableCell>{key}</TableCell>
						<TableCell>
							{#if key === 'osm_node_id'}
								<a
									href="https://www.openstreetmap.org/node/{value}"
									class="underline bold text-blue-400"
									target="_blank"
								>
									{value}
								</a>
							{:else if key === 'osm_way_id'}
								<a
									href="https://www.openstreetmap.org/way/{value}"
									class="underline bold text-blue-400"
									target="_blank"
								>
									{value}
								</a>
							{:else}
								{value}
							{/if}
						</TableCell>
					</TableRow>
				{/each}
			</TableBody>
		</Table>
	{/snippet}

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
			>
				<Popup trigger="click" children={nodeDetails} />
			</Layer>
			<Layer
				id="graph-edge"
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
			>
				<Popup trigger="click" children={nodeDetails} />
			</Layer>
			<Layer
				id="graph-node"
				type="circle"
				filter={['all', ['==', '$type', 'Point']]}
				layout={{}}
				paint={{
					'circle-color': ['match', ['get', 'label'], 'unreachable', '#ff1150', '#11ffaf'],
					'circle-radius': 6
				}}
			>
				<Popup trigger="click" children={nodeDetails} />
			</Layer>
		</GeoJSON>
	{/if}

	{#if start}
		<Marker color="magenta" draggable={false} location={start} />
	{/if}

	{#if destination}
		<Marker color="yellow" draggable={false} location={destination} />
	{/if}
{/if}
