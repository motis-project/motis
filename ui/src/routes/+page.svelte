<script lang="ts">
	import X from 'lucide-svelte/icons/x';
	import maplibregl from 'maplibre-gl';
	import { getStyle } from '$lib/style';
	import Map from '$lib/Map.svelte';
	import Control from '$lib/Control.svelte';
	import GeoJSON from '$lib/GeoJSON.svelte';
	import Layer from '$lib/Layer.svelte';
	import {
		RoutingQuery,
		Location,
		getElevators,
		getFootpaths,
		getGraph,
		getLevels,
		getMatches,
		getPlatforms,
		getRoute,
		Footpaths,
		updateElevator
	} from '$lib/api';
	import { toTable } from '$lib/toTable';
	import {
		Select,
		SelectTrigger,
		SelectValue,
		SelectContent,
		SelectItem
	} from '$lib/components/ui/select';
	import { Toggle } from '$lib/components/ui/toggle';
	import { Button } from '$lib/components/ui/button';
	import {
		Table,
		TableHead,
		TableBody,
		TableCell,
		TableRow,
		TableHeader
	} from '$lib/components/ui/table/index.js';

	let zoom = $state(18);
	let bounds = $state<undefined | maplibregl.LngLatBounds>(undefined);
	let map = $state<null | maplibregl.Map>(null);

	let level = $state(0);
	let levels = $derived.by(async () =>
		bounds === undefined || zoom < 18
			? null
			: await getLevels(maplibregl.LngLatBounds.convert(bounds))
	);

	let showPlatforms = $state(false);
	let platforms = $derived.by(async () =>
		showPlatforms && bounds ? getPlatforms(bounds, level) : null
	);

	let showGraph = $state(false);
	let graph = $state<null | any>(null);
	$effect(async () => {
		graph = showGraph && bounds ? await getGraph(bounds, level) : null;
	});

	let currLevel = 0;
	$effect(() => {
		if (currLevel != level) {
			graph = null;
			currLevel = level;
		}
	});

	let showMatches = $state(false);
	let matches = $state<null | any>(null);
	$effect(async () => {
		matches = showMatches && bounds ? await getMatches(bounds) : null;
	});

	let showElevators = $state<boolean>(false);
	let elevators = $state<null | any>(null);
	$effect(async () => {
		elevators = showElevators && bounds ? await getElevators(bounds) : null;
	});

	let profile = $state({ value: 'foot', label: 'Foot' });
	let start = $state<Location>({
		lat: 50.106847864540164,
		lng: 8.663205312233744,
		level: 0
	});
	let destination = $state<Location>({
		lat: 50.106847864540164,
		lng: 8.665205312233744,
		level: 0
	});
	let query = $derived<RoutingQuery>({
		start,
		destination,
		profile: profile.value,
		direction: 'forward'
	});
	let route = $derived(getRoute(query));

	let footpaths = $state<Footpaths | null>();
	const showLocation = async (props: any) => {
		footpaths = await getFootpaths({ id: props.id, src: props.src });
	};

	let elevator = $state();

	let init = false;
	let startMarker = null;
	let destinationMarker = null;
	$effect(() => {
		if (map && !init) {
			[
				'graph-node',
				'graph-edge',
				'graph-geometry',
				'matches',
				'match',
				'elevators',
				'elevators-match',
				'platform-way',
				'platform-node'
			].forEach((layer) => {
				map!.on('click', layer, async (e) => {
					const props = e.features[0].properties;
					new maplibregl.Popup().setLngLat(e.lngLat).setDOMContent(toTable(props)).addTo(map!);
					e.originalEvent.stopPropagation();

					if (layer === 'matches' && props.type === 'location') {
						await showLocation(props);
					}

					if (layer === 'elevators') {
						elevator = props;
					}
				});

				map!.on('mouseenter', layer, () => {
					map!.getCanvas().style.cursor = 'pointer';
				});

				map!.on('mouseleave', layer, () => {
					map!.getCanvas().style.cursor = '';
				});
			});

			startMarker = new maplibregl.Marker({
				draggable: true,
				color: 'green'
			})
				.setLngLat([start.lng, start.lat])
				.addTo(map)
				.on('dragend', async () => {
					const x = startMarker.getLngLat();
					start.lng = x.lng;
					start.lat = x.lat;
					start.level = level;
				});

			destinationMarker = new maplibregl.Marker({
				draggable: true,
				color: 'red'
			})
				.setLngLat([destination.lng, destination.lat])
				.addTo(map)
				.on('dragend', async () => {
					const x = destinationMarker.getLngLat();
					destination.lng = x.lng;
					destination.lat = x.lat;
					destination.level = level;
				});

			let popup: maplibregl.Popup | null = null;
			map.on('contextmenu', (e) => {
				if (popup != null) {
					popup.remove();
				}
				popup = new maplibregl.Popup({
					anchor: 'top-left'
				});
				const x = e.lngLat;

				const actionsDiv = document.createElement('div');
				const setStart = document.createElement('a');
				setStart.classList.add('m-2');
				setStart.href = '#';
				setStart.innerText = 'start';
				setStart.onclick = () => {
					startMarker.setLngLat(x);
					start.lng = x.lng;
					start.lat = x.lat;
					start.level = level;
					popup!.remove();
				};
				actionsDiv.appendChild(setStart);

				const setDest = document.createElement('a');
				setDest.classList.add('m-2');
				setDest.href = '#';
				setDest.innerText = 'destination';
				setDest.onclick = () => {
					destinationMarker.setLngLat(x);
					destination.lng = x.lng;
					destination.lat = x.lat;
					destination.level = level;
					popup!.remove();
				};
				actionsDiv.appendChild(setDest);

				popup.setLngLat(x).setDOMContent(actionsDiv).addTo(map);
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
	{#if elevator}
		<Control position="bottom-left">
			<div class="bg-white rounded-lg">
				<Button
					variant="ghost"
					on:click={async () => {
						await updateElevator(elevator.id, elevator.status === 'ACTIVE' ? 'INACTIVE' : 'ACTIVE');
					}}
				>
					{#if elevator.status === 'ACTIVE'}
						DEACTIVATE {elevator.id}
					{:else}
						ACTIVATE {elevator.id}
					{/if}
				</Button>
			</div>
		</Control>
	{/if}

	{#if footpaths}
		<Control position="top-left">
			<div class="bg-white rounded-lg">
				<div class="w-full flex justify-between bg-muted shadow-md items-center">
					<h2 class="text-lg ml-2">
						{footpaths.id.name}
						<span class="ml-2 text-sm text-muted-foreground">
							{footpaths.id.id}
						</span>
					</h2>
					<Button
						variant="ghost"
						on:click={() => {
							footpaths = null;
						}}><X /></Button
					>
				</div>
				<div class="h-[500px] overflow-y-scroll">
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
							{#each footpaths.footpaths as f}
								<TableRow>
									<TableCell>{f.id.name}</TableCell>
									<TableCell>{f.default}</TableCell>
									<TableCell>
										<Button
											on:click={async () => {
													start = footpaths!.loc;
													destination = f.loc;
													profile.label = 'Foot';
													profile.value = 'foot';
													startMarker.setLngLat([start.lng, start.lat]);
													destinationMarker.setLngLat([destination.lng, destination.lat]);
													await showLocation(f.id);
												}}
											variant="outline">{f.foot}</Button
										>
									</TableCell>
									<TableCell>
										<Button
											on:click={async () => {
													start = footpaths!.loc;
													destination = f.loc;
													profile.label = 'Wheelchair';
													profile.value = 'wheelchair';
													startMarker.setLngLat([start.lng, start.lat]);
													destinationMarker.setLngLat([destination.lng, destination.lat]);
													await showLocation(f.id);
												}}
											variant="outline">{f.wheelchair}</Button
										>
									</TableCell>
								</TableRow>
							{/each}
						</TableBody>
					</Table>
				</div>
			</div>
		</Control>
	{/if}

	<Control>
		<div class="bg-white rounded-lg">
			<Toggle
				bind:pressed={showGraph}
				variant="outline"
				class={['h-8', 'w-8', showGraph ? 'bg-green-200' : ''].join(' ')}
			>
				G
			</Toggle>
		</div>
	</Control>
	<Control>
		<div class="bg-white rounded-lg">
			<Toggle
				bind:pressed={showMatches}
				variant="outline"
				class="{['h-8', 'w-8', showMatches ? 'bg-green-200' : ''].join(' ')}}"
			>
				M
			</Toggle>
		</div>
	</Control>
	<Control>
		<div class="bg-white rounded-lg">
			<Toggle
				bind:pressed={showPlatforms}
				variant="outline"
				class={['h-8', 'w-8', showPlatforms ? 'bg-green-200' : ''].join(' ')}
			>
				P
			</Toggle>
		</div>
	</Control>
	<Control>
		<div class="bg-white rounded-lg">
			<Toggle
				bind:pressed={showElevators}
				variant="outline"
				class={['h-8', 'w-8', showElevators ? 'bg-green-200' : ''].join(' ')}
			>
				E
			</Toggle>
		</div>
	</Control>
	<Control>
		<div class="bg-white rounded-lg">
			<Select bind:selected={profile}>
				<SelectTrigger>
					<SelectValue placeholder="Theme" />
				</SelectTrigger>
				<SelectContent>
					<SelectItem value="wheelchair">Wheelchair</SelectItem>
					<SelectItem value="foot">Foot</SelectItem>
					<SelectItem value="bike">Bike</SelectItem>
					<SelectItem value="car">Car</SelectItem>
				</SelectContent>
			</Select>
		</div>
	</Control>

	{#if levels}
		{#await levels then lvls}
			{#each lvls as l}
				<Control>
					<div class="bg-white rounded-lg">
						<Toggle
							variant="outline"
							class={['h-8', 'w-8', level == l ? 'bg-green-200' : ''].join(' ')}
							onclick={() => {
								level = l;
							}}
						>
							{l}
						</Toggle>
					</div>
				</Control>
			{/each}
		{/await}
	{/if}

	{#await platforms then p}
		<GeoJSON id="platforms" data={p}>
			<Layer
				id="platform-way"
				type="line"
				layout={{
					'line-join': 'round',
					'line-cap': 'round'
				}}
				filter={['all', ['==', 'type', 'way'], ['any', ['!has', 'level'], ['==', 'level', level]]]}
				paint={{
					'line-color': '#00d924',
					'line-width': 3
				}}
			/>
			<Layer
				id="platform-node"
				type="circle"
				layout={{}}
				filter={['all', ['==', 'type', 'node'], ['any', ['!has', 'level'], ['==', 'level', level]]]}
				paint={{
					'circle-color': '#0700d9',
					'circle-radius': 5
				}}
			/>
		</GeoJSON>
	{/await}

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
						'line-color': '#1966a4',
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
						'line-color': '#42a5f5',
						'line-width': 5,
						'line-opacity': 0.8
					}}
				/>
			</GeoJSON>
		{/if}
	{/await}

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
			/>
			<Layer
				id="graph-node"
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

	{#if elevators != null}
		<GeoJSON id="elevators" data={elevators}>
			<Layer
				id="elevators"
				type="circle"
				filter={['all', ['==', '$type', 'Point']]}
				layout={{}}
				paint={{
					'circle-color': ['match', ['get', 'status'], 'ACTIVE', '#32a852', '#ff144b'],
					'circle-radius': 5
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
			/>
		</GeoJSON>
	{/if}
</Map>
