<script lang="ts">
	import X from 'lucide-svelte/icons/x';
	import { getStyle } from '$lib/map/style';
	import Map from '$lib/map/Map.svelte';
	import Control from '$lib/map/Control.svelte';
	import SearchMask from './SearchMask.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { Card } from '$lib/components/ui/card';
	import { type Itinerary, plan, type PlanResponse, trip } from '$lib/openapi';
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
	import { toDateTime } from '$lib/toDateTime';

	const urlParams = browser && new URLSearchParams(window.location.search);
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

	let level = $state(0);
	let zoom = $state(15);
	let bounds = $state<maplibregl.LngLatBoundsLike>();
	let map = $state<maplibregl.Map>();

	let fromMarker = $state<maplibregl.Marker>();
	let toMarker = $state<maplibregl.Marker>();
	let from = $state<Location>({ label: '', value: {} });
	let to = $state<Location>({ label: '', value: {} });
	let dateTime = $state<Date>(new Date());
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
						date: toDateTime(dateTime)[0],
						time: toDateTime(dateTime)[1],
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

	const onClickTrip = (tripId: string, date: string) => {
		trip({ query: { tripId, date } }).then((r) => {
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
			return { url: `${client.getConfig().baseUrl}/${url}` };
		}
		if (url.startsWith('/')) {
			return { url: `${client.getConfig().baseUrl}/tiles${url}` };
		}
	}}
	center={[8.652235, 49.876908]}
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
			<SearchMask bind:from bind:to bind:dateTime bind:timeType bind:wheelchair {theme} />
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

	<Popup trigger="contextmenu" children={contextMenu} />

	{#if from}
		<Marker color="green" draggable={true} bind:location={from} bind:marker={fromMarker} />
	{/if}

	{#if to}
		<Marker color="red" draggable={true} bind:location={to} bind:marker={toMarker} />
	{/if}
</Map>
