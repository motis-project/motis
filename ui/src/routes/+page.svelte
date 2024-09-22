<script lang="ts">
	import X from 'lucide-svelte/icons/x';
	import LoaderCircle from 'lucide-svelte/icons/loader-circle';
	import maplibregl from 'maplibre-gl';
	import { getStyle } from '$lib/style';
	import Map from '$lib/Map.svelte';
	import Control from '$lib/Control.svelte';
	import GeoJSON from '$lib/GeoJSON.svelte';
	import Layer from '$lib/Layer.svelte';
	import { Button } from '$lib/components/ui/button/index.js';
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
		updateElevator,
		type Elevator
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
	import {
		Table,
		TableHead,
		TableBody,
		TableCell,
		TableRow,
		TableHeader
	} from '$lib/components/ui/table/index.js';
	import { Label } from '$lib/components/ui/label/index.js';
	import { plan, type Itinerary, type Match, type PlanResponse } from '$lib/openapi';
	import { Separator } from '$lib/components/ui/separator';
	import { Card } from '$lib/components/ui/card';
	import ConnectionDetail from './ConnectionDetail.svelte';
	import Time from './Time.svelte';
	import { routeColor } from '$lib/modeStyle';
	import { getModeStyle } from '$lib/modeStyle';
	import { itineraryToGeoJSON } from '$lib/ItineraryToGeoJSON';
	import { formatDurationSec } from '$lib/formatDuration';
	import DateInput from '$lib/DateInput.svelte';
	import * as RadioGroup from '$lib/components/ui/radio-group/index.js';
	import ComboBox from '$lib/ComboBox.svelte';
	import { type Selected } from 'bits-ui';

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
	let platforms = $state<null | any>(null);
	$effect(async () => {
		platforms = showPlatforms && bounds ? getPlatforms(bounds, level) : null;
	});

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
		lat: 49.872584079,
		lng: 8.6312708899,
		level: 0
	});
	let destination = $state<Location>({
		lat: 50.11352164499803,
		lng: 8.677728968355844,
		level: 0
	});
	let query = $derived<RoutingQuery>({
		start,
		destination,
		profile: profile.value,
		direction: 'forward'
	});
	let footRoute = $derived(getRoute(query));

	let footpaths = $state<Footpaths | null>(null);
	const showLocation = async (props: any) => {
		footpaths = await getFootpaths({ id: props.id, src: props.src });
	};

	let elevator = $state<Elevator | null>();
	let elevatorUpdate = $state<Promise<Response> | null>(null);

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
						const e = elevators.features.find((f) => f.properties.id === props.id).properties;
						elevator = {
							...e,
							outOfService: e.outOfService.map(([from, to]) => [new Date(from), new Date(to)])
						};
						console.log('elevator: ', elevator);
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

	const pad = (x: number) => ('0' + x).slice(-2);
	let timeType = $state('departure');
	let dateTime = $state(new Date());
	let arriveBy = $derived(timeType === 'arrival');
	let baseQuery = $derived({
		date: `${pad(dateTime.getUTCMonth() + 1)}-${pad(dateTime.getUTCDate())}-${dateTime.getUTCFullYear()}`,
		time: `${pad(dateTime.getUTCHours())}:${pad(dateTime.getUTCMinutes())}`,
		fromPlace: `${query.start.lat},${query.start.lng},${query.start.level}`,
		toPlace: `${query.destination.lat},${query.destination.lng},${query.destination.level}`,
		wheelchair: profile.value == 'wheelchair',
		timetableView: true,
		arriveBy
	});

	let routingResponses = $state<Array<Promise<PlanResponse>>>([]);

	$effect(() => {
		routingResponses = [plan<true>({ query: baseQuery }).then((response) => response.data)];
	});

	let itinerary = $state<Itinerary | null>(null);
	let route = $derived(itineraryToGeoJSON(itinerary));

	// client ID: a9b1f1ad1051790a9c6970db85710986
	// client Secret: df987129855de70a804f146718aac956
	// client Secret: 30dee8771d325304274b7c2555fae33e
</script>

<Map
	bind:map
	bind:bounds
	transformRequest={(url, _resourceType) => {
		if (url.startsWith('/')) {
			return { url: `https://europe.motis-project.de/tiles${url}` };
		}
	}}
	center={[8.563351200419433, 50]}
	zoom={10}
	class="h-screen"
	style={getStyle(level)}
>
	{#if elevator}
		<Control position="bottom-left">
			<Card class="flex flex-col">
				<div class="w-full flex justify-between bg-muted items-center">
					<h2 class="text-lg ml-2 font-bold">
						Fahrstuhl {elevator.desc}
						<span class="ml-2 text-sm text-muted-foreground">
							{elevator.id}
						</span>
					</h2>
					<Button
						variant="ghost"
						on:click={() => {
							elevator = null;
						}}
					>
						<X />
					</Button>
				</div>
				<Table>
					<TableHeader>
						<TableRow>
							<TableHead class="font-semibold">Nicht verfügbar von</TableHead>
							<TableHead class="font-semibold">bis</TableHead>
							<TableHead></TableHead>
						</TableRow>
					</TableHeader>
					<TableBody>
						{#each elevator.outOfService as _, i}
							<TableRow>
								<TableCell>
									<DateInput bind:value={elevator.outOfService[i][0]} />
								</TableCell>
								<TableCell>
									<DateInput bind:value={elevator.outOfService[i][1]} />
								</TableCell>
								<TableCell>
									<Button
										variant="outline"
										on:click={() => {
											elevator!.outOfService.splice(i, 1);
										}}
									>
										<X />
									</Button>
								</TableCell>
							</TableRow>
						{/each}
					</TableBody>
				</Table>
				<div class="flex space-x-2 m-2 justify-end">
					<Button
						variant="outline"
						on:click={() => {
							elevator!.outOfService.push([new Date(), new Date()]);
						}}
					>
						Wartungsfenster hinzufügen
					</Button>
					<Button
						class="w-48"
						variant="outline"
						on:click={async () => {
							elevatorUpdate = updateElevator(elevator!);
						}}
					>
						{#if elevatorUpdate != null}
							{#await elevatorUpdate}
								<LoaderCircle class="animate-spin w-4 h-4" />
							{:then _}
								Aktualisieren
							{/await}
						{:else}
							Aktualisieren
						{/if}
					</Button>
				</div>
			</Card>
		</Control>
	{/if}

	<Control position="bottom-left">
		<Card class="w-[520px] max-h-[90vh] overflow-y-auto overflow-x-hidden bg-white rounded-lg">
			{#if itinerary !== null}
				<div class="w-full flex justify-between bg-muted items-center">
					<h2 class="text-lg ml-2 font-bold">Journey Details</h2>
					<Button
						variant="ghost"
						on:click={() => {
							itinerary = null;
						}}
					>
						<X />
					</Button>
				</div>
				<div class="p-4">
					<ConnectionDetail {itinerary} />
				</div>
			{/if}

			<div class="flex flex-col w-full" class:hidden={itinerary}>
				<div class="flex flex-col space-y-4 p-4 shadow-md rounded">
					<ComboBox
						name="from"
						inputClass="w-full bg-white"
						placeholder="From"
						onSelectedChange={(match: Selected<Match> | undefined) => {
							if (match) {
								start.lng = match.value.lon;
								start.lat = match.value.lat;
								startMarker.setLngLat([start.lng, start.lat]);
							}
						}}
					/>
					<ComboBox
						name="to"
						inputClass="w-full bg-white"
						placeholder="To"
						onSelectedChange={(match: Selected<Match> | undefined) => {
							if (match) {
								destination.lng = match.value.lon;
								destination.lat = match.value.lat;
								destinationMarker.setLngLat([destination.lng, destination.lat]);
							}
						}}
					/>
					<div class="flex flex-row space-x-4 justify-between">
						<div class="flex">
							<DateInput class="bg-white" bind:value={dateTime} />
							<RadioGroup.Root class="flex space-x-1 ml-1" bind:value={timeType}>
								<Label
									for="departure"
									class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
								>
									<RadioGroup.Item
										value="departure"
										id="departure"
										class="sr-only"
										aria-label="Abfahrt"
									/>
									<span>Abfahrt</span>
								</Label>
								<Label
									for="arrival"
									class="flex items-center rounded-md border-2 border-muted bg-popover p-1 px-2 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
								>
									<RadioGroup.Item
										value="arrival"
										id="arrival"
										class="sr-only"
										aria-label="Ankunft"
									/>
									<span>Ankunft</span>
								</Label>
							</RadioGroup.Root>
						</div>
						<div class="min-w-24">
							<Select bind:selected={profile}>
								<SelectTrigger class="bg-white">
									<SelectValue placeholder="Profile" />
								</SelectTrigger>
								<SelectContent>
									<SelectItem value="wheelchair">Wheelchair</SelectItem>
									<SelectItem value="foot">Foot</SelectItem>
									<SelectItem value="bike">Bike</SelectItem>
									<SelectItem value="car">Car</SelectItem>
								</SelectContent>
							</Select>
						</div>
					</div>
				</div>
				<div class="flex flex-col space-y-8 h-[45vh] overflow-y-auto px-4 py-8">
					{#each routingResponses as routingResponse, rI}
						{#await routingResponse}
							<div class="flex items-center justify-center w-full">
								<LoaderCircle class="animate-spin w-12 h-12 m-20" />
							</div>
						{:then r}
							{#if rI === 0}
								<div class="w-full flex justify-between items-center space-x-4">
									<div class="border-t w-full h-0"></div>
									<button
										onclick={() => {
											routingResponses.splice(
												0,
												0,
												plan({ query: { ...baseQuery, pageCursor: r.previousPageCursor } }).then(
													(x) => x.data!
												)
											);
										}}
										class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white font-bold border rounded-lg"
									>
										früher
									</button>
									<div class="border-t w-full h-0"></div>
								</div>
							{/if}
							{#each r.itineraries as it, i}
								{@const date = new Date(it.startTime).toLocaleDateString()}
								{@const predDate = new Date(
									r.itineraries[i == 0 ? 0 : i - 1].startTime
								).toLocaleDateString()}
								<a
									href="#"
									onclick={() => {
										itinerary = it;
									}}
								>
									<Card class="p-4">
										<div class="text-base h-8 flex justify-between items-center space-x-4 w-full">
											<div>
												<div class="text-xs font-bold uppercase text-slate-400">Departure Time</div>
												<Time timestamp={it.startTime} />
											</div>
											<Separator orientation="vertical" />
											<div>
												<div class="text-xs font-bold uppercase text-slate-400">Arrival Time</div>
												<Time timestamp={it.endTime} />
											</div>
											<Separator orientation="vertical" />
											<div>
												<div class="text-xs font-bold uppercase text-slate-400">Transfers</div>
												<div class="flex justify-center w-full">{it.transfers}</div>
											</div>
											<Separator orientation="vertical" />
											<div>
												<div class="text-xs font-bold uppercase text-slate-400">Duration</div>
												<div class="flex justify-center w-full">
													{formatDurationSec(it.duration)}
												</div>
											</div>
										</div>
										<Separator class="my-2" />
										<div class="mt-4 flex space-x-4">
											{#each it.legs.filter((l) => l.routeShortName) as l}
												<div
													class="flex items-center py-1 px-2 rounded-lg font-bold"
													style={routeColor(l)}
												>
													<svg class="relative mr-1 w-4 h-4 fill-white rounded-full">
														<use xlink:href={`#${getModeStyle(l.mode)[0]}`}></use>
													</svg>
													{l.routeShortName}
												</div>
											{/each}
										</div>
									</Card>
								</a>
							{/each}
							{#if rI === routingResponses.length - 1}
								<div class="w-full flex justify-between items-center space-x-4">
									<div class="border-t w-full h-0"></div>
									<button
										onclick={() => {
											routingResponses.push(
												plan({ query: { ...baseQuery, pageCursor: r.nextPageCursor } }).then(
													(x) => x.data!
												)
											);
										}}
										class="px-2 py-1 bg-blue-600 hover:!bg-blue-700 text-white font-bold border rounded-lg"
									>
										später
									</button>
									<div class="border-t w-full h-0"></div>
								</div>
							{/if}
						{:catch e}
							<div>Error: {e}</div>
						{/await}
					{/each}
				</div>
			</div>
		</Card>
	</Control>

	{#if footpaths}
		<Control position="top-left">
			<div class="h-[45vh] bg-white rounded-lg">
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
				<div class="h-full bg-white overflow-y-scroll">
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
									<TableCell>
										{#if f.default !== undefined}
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
										{/if}
									</TableCell>
									<TableCell>
										{#if f.foot !== undefined}
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
										{/if}
									</TableCell>
									<TableCell>
										{#if f.wheelchair !== undefined}
											<Button
												class={f.wheelchair_uses_elevator ? 'text-red-500' : 'text-green-500'}
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
										{/if}
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

	{#if platforms !== null}
		{#await platforms then p}
			<GeoJSON id="platforms" data={p}>
				<Layer
					id="platform-way"
					type="line"
					layout={{
						'line-join': 'round',
						'line-cap': 'round'
					}}
					filter={[
						'all',
						['==', 'type', 'way'],
						['any', ['!has', 'level'], ['==', 'level', level]]
					]}
					paint={{
						'line-color': '#00d924',
						'line-width': 3
					}}
				/>
				<Layer
					id="platform-node"
					type="circle"
					layout={{}}
					filter={[
						'all',
						['==', 'type', 'node'],
						['any', ['!has', 'level'], ['==', 'level', level]]
					]}
					paint={{
						'circle-color': '#0700d9',
						'circle-radius': 5
					}}
				/>
			</GeoJSON>
		{/await}
	{/if}

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
						'line-color': ['get', 'outlineColor'],
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
						'line-color': ['get', 'color'],
						'line-width': 5,
						'line-opacity': 0.8
					}}
				/>
			</GeoJSON>
		{/if}
	{/await}

	<Control position="bottom-right">
		<div class="bg-black/50 p-4 rounded-md text-center font-bold text-lg text-white">
			{#await footRoute}
				Loading...
			{:then r}
				{#if r.metadata}
					{formatDurationSec(r.metadata.duration)}
					<br />
					{Math.round(r.metadata.distance)} m
					{#if r.metadata.uses_elevator}
						<div>mit Fahrstuhl</div>
					{/if}
				{:else}
					No foot route found.
				{/if}
			{/await}
		</div>
	</Control>

	{#await footRoute then r}
		{#if itinerary == null && r.type == 'FeatureCollection'}
			<GeoJSON id="foot-route" data={r}>
				<Layer
					id="foot-route-path-outline"
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
					id="foot-route-path"
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
