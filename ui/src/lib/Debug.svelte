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
	import * as Card from '$lib/components/ui/card';
	import Marker from '$lib/map/Marker.svelte';
	import { posToLocation, type Location as ApiLocation } from '$lib/Location';
	import geojson from 'geojson';
	import Popup from '$lib/map/Popup.svelte';
	import { client } from '$lib/openapi';
	import X from 'lucide-svelte/icons/x';
	import LoaderCircle from 'lucide-svelte/icons/loader-circle';
	import DateInput from './DateInput.svelte';

	const baseUrl = client.getConfig().baseUrl;

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

	type ElevatorStatus = 'ACTIVE' | 'INACTIVE';

	type Elevator = {
		id: number;
		status: ElevatorStatus;
		desc: string;
		outOfService: [Date, Date][];
	};

	const toLocation = (l: ApiLocation): Location => {
		return {
			lat: l.value.match!.lat,
			lng: l.value.match!.lon,
			level: l.value.match!.level ?? 0
		};
	};

	const getRoute = async (query: RoutingQuery) => {
		return await post('/api/route', query);
	};

	const getMatches = async (bounds: maplibregl.LngLatBounds) => {
		return await post('/api/matches', bounds.toArray().flat());
	};

	const getElevators = async (bounds: maplibregl.LngLatBounds) => {
		return await post('/api/elevators', bounds.toArray().flat());
	};

	const updateElevator = async (e: { id: number; status: ElevatorStatus }) => {
		console.log(JSON.stringify(e));
		return await post('/api/update_elevator', e);
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

	const parseElevator = (e: { outOfService: string }) => {
		return {
			...e,
			outOfService: JSON.parse(e.outOfService).map(([from, to]: [string, string]) => {
				return [new Date(from), new Date(to)];
			})
		};
	};

	let graph = $state<null | geojson.GeoJSON>(null);
	let elevators = $state<null | geojson.GeoJSON>(null);
	$effect(() => {
		if (debug && bounds) {
			getGraph(maplibregl.LngLatBounds.convert(bounds), level).then((response: geojson.GeoJSON) => {
				graph = response;
			});
			getElevators(maplibregl.LngLatBounds.convert(bounds)).then((response: geojson.GeoJSON) => {
				elevators = response;
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

	let elevatorUpdate = $state<Promise<Response> | null>(null);
	let elevator = $state<Elevator | null>(null);
</script>

<Button
	size="icon"
	variant={debug ? 'default' : 'outline'}
	onclick={() => {
		debug = !debug;
	}}
>
	<Bug size="icon" class="h-[1.2rem] w-[1.2rem]" />
</Button>

<!-- eslint-disable-next-line -->
{#snippet propertiesTable(_1: maplibregl.MapMouseEvent, _2: () => void, features: any)}
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

{#if elevator}
	<Control position="bottom-right">
		<Card.Root class="min-w-[550px] mb-4">
			<div class="w-full flex justify-between bg-muted items-center">
				<h2 class="text-lg ml-2 font-bold">
					Fahrstuhl {elevator.desc}
					<span class="ml-2 text-sm text-muted-foreground">
						{elevator.id}
					</span>
				</h2>
				<Button variant="ghost" onclick={() => (elevator = null)}>
					<X />
				</Button>
			</div>

			<Card.Content class="flex flex-col gap-6">
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead class="font-semibold">Not available from</TableHead>
							<TableHead class="font-semibold">to</TableHead>
							<TableHead></TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{#if elevator.outOfService}
							{#each elevator.outOfService as _, i}
								<TableRow>
									<TableCell>
										<DateInput bind:value={elevator.outOfService[i][0]} />
									</TableCell>
									<TableCell>
										<DateInput bind:value={elevator.outOfService[i][1]} />
									</TableCell>
									<TableCell>
										<Button variant="outline" onclick={() => elevator!.outOfService!.splice(i, 1)}>
											<X />
										</Button>
									</TableCell>
								</TableRow>
							{/each}
						{/if}
					</TableBody>
				</Table>

				<div class="flex justify-between gap-4">
					<Button
						variant="outline"
						onclick={() => elevator!.outOfService.push([new Date(), new Date()])}
					>
						Add Maintainance
					</Button>
					<Button
						class="w-48"
						variant="outline"
						onclick={() => {
							elevatorUpdate = updateElevator(elevator!);
						}}
					>
						{#if elevatorUpdate != null}
							{#await elevatorUpdate}
								<LoaderCircle class="animate-spin w-4 h-4" />
							{:then _}
								Update
							{/await}
						{:else}
							Update
						{/if}
					</Button>

					<Button
						onclick={() => {
							elevatorUpdate = updateElevator({
								id: elevator!.id,
								status: elevator!.status === 'ACTIVE' ? 'INACTIVE' : 'ACTIVE'
							});
						}}
					>
						{#if elevator.status === 'ACTIVE'}
							DEACTIVATE
						{:else}
							ACTIVATE
						{/if}
					</Button>
				</div>
			</Card.Content>
		</Card.Root>
	</Control>
{/if}

{#if debug}
	{#if fps}
		{#await fps then f}
			{#if f}
				<Control position="bottom-right">
					<Card.Root class="w-[600px] h-[500px] overflow-y-auto bg-background rounded-lg">
						<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
							<h2 class="ml-2 text-base font-semibold">
								{f.place.name}
								{f.place.track}
								<span class="text-sm text-muted-foreground">Level: {f.place.level}</span>
							</h2>
							<Button
								variant="ghost"
								onclick={() => {
									id = undefined;
								}}
							>
								<X />
							</Button>
						</div>
						<Table>
							<TableHeader>
								<TableRow>
									<TableHead class="text-center">Station</TableHead>
									<TableHead class="text-center">Default</TableHead>
									<TableHead class="text-center">Foot</TableHead>
									<TableHead class="text-center">Foot Routed</TableHead>
									<TableHead class="text-center">Wheelchair</TableHead>
									<TableHead class="text-center">Wheelchair Routed</TableHead>
								</TableRow>
							</TableHeader>
							<TableBody>
								{#each f.footpaths as x}
									<TableRow>
										<TableCell>
											{x.to.name} <br />
											<span class="text-xs text-muted-foreground font-mono">
												{x.to.stopId}
											</span>
										</TableCell>
										<TableCell>
											{#if x.default !== undefined}
												<Button
													variant="outline"
													onclick={() => {
														start = posToLocation(f.place, f.place.level);
														destination = posToLocation(x.to, x.to.level);
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
													onclick={() => {
														start = posToLocation(f.place, f.place.level);
														destination = posToLocation(x.to, x.to.level);
														profile = 'foot';
													}}
												>
													{x.foot}
												</Button>
											{/if}
										</TableCell>
										<TableCell>
											{#if x.footRouted !== undefined}
												<Button
													variant="outline"
													onclick={() => {
														start = posToLocation(f.place, f.place.level);
														destination = posToLocation(x.to, x.to.level);
														profile = 'foot';
													}}
												>
													{x.footRouted}
												</Button>
											{/if}
										</TableCell>
										<TableCell>
											{#if x.wheelchair !== undefined}
												<Button
													class={x.wheelchairUsesElevator ? 'text-red-500' : 'text-green-500'}
													variant="outline"
													onclick={() => {
														start = posToLocation(f.place, f.place.level);
														destination = posToLocation(x.to, x.to.level);
														profile = 'wheelchair';
													}}
												>
													{x.wheelchair}
												</Button>
											{/if}
										</TableCell>
										<TableCell>
											{#if x.wheelchairRouted !== undefined}
												<Button
													class={x.wheelchairUsesElevator ? 'text-red-500' : 'text-green-500'}
													variant="outline"
													onclick={() => {
														start = posToLocation(f.place, f.place.level);
														destination = posToLocation(x.to, x.to.level);
														profile = 'wheelchair';
													}}
												>
													{x.wheelchairRouted}
												</Button>
											{/if}
										</TableCell>
									</TableRow>
								{/each}
							</TableBody>
						</Table>
					</Card.Root>
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
				>
					<Popup trigger="click" children={propertiesTable} />
				</Layer>
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
				>
					<Popup trigger="click" children={propertiesTable} />
				</Layer>
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
				<Popup trigger="click" children={propertiesTable} />
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
				<Popup trigger="click" children={propertiesTable} />
			</Layer>
			<Layer
				id="graph-node"
				type="circle"
				filter={['all', ['==', '$type', 'Point']]}
				layout={{}}
				paint={{
					'circle-color': ['match', ['get', 'label'], 'unreachable', '#ff1150', '#11ffaf'],
					'circle-radius': 5
				}}
			>
				<Popup trigger="click" children={propertiesTable} />
			</Layer>
		</GeoJSON>
	{/if}

	{#if elevators}
		<GeoJSON id="elevators" data={elevators}>
			<Layer
				id="elevators"
				type="circle"
				filter={['all', ['==', '$type', 'Point']]}
				layout={{}}
				paint={{
					'circle-color': ['match', ['get', 'status'], 'ACTIVE', '#ffff00', '#ff00ff'],
					'circle-radius': 8
				}}
				onclick={(e) => {
					// @ts-expect-error type mismatch
					elevator = parseElevator(e.features![0].properties);
				}}
			/>
			<Layer
				id="elevators-match"
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
			>
				<Popup trigger="click" children={propertiesTable} />
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
