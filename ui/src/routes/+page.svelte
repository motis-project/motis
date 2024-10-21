<script lang="ts">
	import X from 'lucide-svelte/icons/x';
	import { getStyle } from '$lib/map/style';
	import Map from '$lib/map/Map.svelte';
	import Control from '$lib/map/Control.svelte';
	import SearchMask from './SearchMask.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { Card } from '$lib/components/ui/card';
	import { initial, type Itinerary, type Match, plan, type PlanResponse, trip } from '$lib/openapi';
	import ItineraryList from './ItineraryList.svelte';
	import ConnectionDetail from './ConnectionDetail.svelte';
	import { Button } from '$lib/components/ui/button';
	import ItineraryGeoJson from './ItineraryGeoJSON.svelte';
	import maplibregl from 'maplibre-gl';
	import { browser } from '$app/environment';
	import { cn } from '$lib/utils';
	import Debug from './Debug.svelte';
	import Marker from '$lib/map/Marker.svelte';
	import Popup from '$lib/map/Popup.svelte';
	import LevelSelect from './LevelSelect.svelte';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { client } from '$lib/openapi';
	import StopTimes from './StopTimes.svelte';
	import { onMount } from 'svelte';
	import RailViz from './RailViz.svelte';

	const urlParams = browser ? new URLSearchParams(window.location.search) : undefined;
	const hasDebug = urlParams && urlParams.has('debug');
	const hasDark = urlParams && urlParams.has('dark');

	let theme: 'light' | 'dark' =
		(hasDark ? 'dark' : undefined) ??
		(browser && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
			? 'dark'
			: 'light');
	if (theme === 'dark') {
		document.documentElement.classList.add('dark');
	}

	let center = $state.raw<[number, number]>([8.652235, 49.876908]);
	let level = $state(0);
	let zoom = $state(15);
	let bounds = $state<maplibregl.LngLatBoundsLike>();
	let map = $state<maplibregl.Map>();

	onMount(() => {
		initial().then((d) => {
			const r = d.data;
			if (r) {
				center = [r.lon, r.lat];
				zoom = r.zoom;
			}
		});
	});

	let fromParam: Match | undefined = undefined;
	let toParam: Match | undefined = undefined;
	if (browser && urlParams && urlParams.has('from') && urlParams.has('to')) {
		fromParam = JSON.parse(urlParams.get('from') ?? '') ?? {};
		toParam = JSON.parse(urlParams.get('to') ?? '') ?? {};
	}

	let fromMatch = {
		match: fromParam
	};
	let toMatch = {
		match: toParam
	};

	let fromMarker = $state<maplibregl.Marker>();
	let toMarker = $state<maplibregl.Marker>();
	let from = $state<Location>({
		label: fromParam ? fromParam['name'] : '',
		value: fromParam ? fromMatch : {}
	});
	let to = $state<Location>({
		label: toParam ? toParam['name'] : '',
		value: toParam ? toMatch : {}
	});
	let time = $state<Date>(new Date());
	let timeType = $state<string>('departure');
	let wheelchair = $state(false);

	const toPlaceString = (l: Location) => {
		if (l.value.match?.type === 'STOP') {
			return l.value.match.id;
		} else if (l.value.match?.level) {
			return `${lngLatToStr(l.value.match!)},${l.value.match.level}`;
		} else {
			return `${lngLatToStr(l.value.match!)},0`;
		}
	};
	let baseQuery = $derived(
		from.value.match && to.value.match
			? {
					query: {
						time: time.toISOString(),
						fromPlace: toPlaceString(from),
						toPlace: toPlaceString(to),
						arriveBy: timeType === 'arrival',
						timetableView: true,
						wheelchair
					}
				}
			: undefined
	);
	let routingResponses = $state<Array<Promise<PlanResponse>>>([]);
	$effect(() => {
		if (baseQuery) {
			routingResponses = [plan<true>(baseQuery).then((response) => response.data)];
			selectedItinerary = undefined;
			selectedStop = undefined;
		}
	});

	let selectedItinerary = $state<Itinerary>();
	$effect(() => {
		if (selectedItinerary && map) {
			const start = maplibregl.LngLat.convert(selectedItinerary.legs[0].from);
			const box = new maplibregl.LngLatBounds(start, start);
			selectedItinerary.legs.forEach((l) => {
				box.extend(l.from);
				box.extend(l.to);
				l.intermediateStops?.forEach((x) => {
					box.extend(x);
				});
			});
			const padding = { top: 96, right: 96, bottom: 96, left: 640 };
			map.flyTo({ ...map.cameraForBounds(box), padding });
		}
	});

	let stopArriveBy = $state<boolean>();
	let selectedStop = $state<{ name: string; stopId: string; time: Date }>();

	const onClickTrip = (tripId: string) => {
		trip({ query: { tripId } }).then((r) => {
			selectedItinerary = r.data;
			selectedStop = undefined;
		});
	};

	type CloseFn = () => void;
</script>

{#snippet contextMenu(e: maplibregl.MapMouseEvent, close: CloseFn)}
	<Button
		variant="outline"
		on:click={() => {
			from = posToLocation(e.lngLat);
			fromMarker?.setLngLat(from.value.match!);
			close();
		}}
	>
		From
	</Button>
	<Button
		variant="outline"
		on:click={() => {
			to = posToLocation(e.lngLat);
			toMarker?.setLngLat(to.value.match!);
			close();
		}}
	>
		To
	</Button>
{/snippet}

<Map
	bind:map
	bind:bounds
	bind:zoom
	transformRequest={(url: string) => {
		if (url.startsWith('/sprite')) {
			return { url: `${window.location.origin}${url}` };
		}
		if (url.startsWith('/')) {
			return { url: `${client.getConfig().baseUrl}/tiles${url}` };
		}
	}}
	{center}
	class={cn('h-screen overflow-clip', theme)}
	style={getStyle(theme, level)}
>
	{#if hasDebug}
		<Control position="top-right">
			<Debug {bounds} {level} />
		</Control>
	{/if}

	<Control position="top-left">
		<Card class="w-[500px] overflow-y-auto overflow-x-hidden bg-background rounded-lg">
			<SearchMask bind:from bind:to bind:time bind:timeType bind:wheelchair {theme} />
		</Card>
	</Control>

	<LevelSelect {bounds} {zoom} bind:level />

	{#if !selectedItinerary && baseQuery && routingResponses.length !== 0}
		<Control position="top-left">
			<Card
				class="w-[500px] max-h-[70vh] overflow-y-auto overflow-x-hidden bg-background rounded-lg"
			>
				<ItineraryList {routingResponses} {baseQuery} bind:selectedItinerary />
			</Card>
		</Control>
	{/if}

	{#if selectedItinerary && !selectedStop}
		<Control position="top-left">
			<Card class="w-[500px] bg-background rounded-lg">
				<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
					<h2 class="ml-2 text-base font-semibold">Journey Details</h2>
					<Button
						variant="ghost"
						on:click={() => {
							selectedItinerary = undefined;
						}}
					>
						<X />
					</Button>
				</div>
				<div class="p-4 overflow-y-auto overflow-x-hidden max-h-[70vh]">
					<ConnectionDetail
						itinerary={selectedItinerary}
						onClickStop={(name: string, stopId: string, time: Date) => {
							stopArriveBy = false;
							selectedStop = { name, stopId, time };
						}}
						{onClickTrip}
					/>
				</div>
			</Card>
		</Control>
		<ItineraryGeoJson itinerary={selectedItinerary} {level} />
	{/if}

	{#if selectedStop}
		<Control position="top-left">
			<Card class="w-[500px] bg-background rounded-lg">
				<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
					<h2 class="ml-2 text-base font-semibold">
						{#if stopArriveBy}
							Ankünfte
						{:else}
							Abfahrten
						{/if}
						in
						{selectedStop.name}
					</h2>
					<Button
						variant="ghost"
						on:click={() => {
							selectedStop = undefined;
						}}
					>
						<X />
					</Button>
				</div>
				<div class="p-6 overflow-y-auto overflow-x-hidden max-h-[70vh]">
					<StopTimes
						stopId={selectedStop.stopId}
						time={selectedStop.time}
						bind:arriveBy={stopArriveBy}
						{onClickTrip}
					/>
				</div>
			</Card>
		</Control>
	{/if}

	<RailViz {map} {bounds} {zoom} {onClickTrip} />

	<Popup trigger="contextmenu" children={contextMenu} />

	{#if from}
		<Marker color="green" draggable={true} bind:location={from} bind:marker={fromMarker} />
	{/if}

	{#if to}
		<Marker color="red" draggable={true} bind:location={to} bind:marker={toMarker} />
	{/if}
</Map>
