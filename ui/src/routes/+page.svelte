<script lang="ts">
	import X from 'lucide-svelte/icons/x';
	import { getStyle } from '$lib/map/style';
	import Map from '$lib/map/Map.svelte';
	import Control from '$lib/map/Control.svelte';
	import SearchMask from '$lib/SearchMask.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { Card } from '$lib/components/ui/card';
	import {
		initial,
		type Match,
		plan,
		type ElevationCosts,
		type PlanResponse,
		type Itinerary,
		type Mode,
		type PlanData
	} from '$lib/api/openapi';
	import ItineraryList from '$lib/ItineraryList.svelte';
	import ConnectionDetail from '$lib/ConnectionDetail.svelte';
	import { Button } from '$lib/components/ui/button';
	import ItineraryGeoJson from '$lib/ItineraryGeoJSON.svelte';
	import maplibregl from 'maplibre-gl';
	import { browser } from '$app/environment';
	import {
		closeItinerary,
		cn,
		onClickStop,
		onClickTrip,
		pushStateWithQueryString
	} from '$lib/utils';
	import Debug from '$lib/Debug.svelte';
	import Marker from '$lib/map/Marker.svelte';
	import Popup from '$lib/map/Popup.svelte';
	import LevelSelect from '$lib/LevelSelect.svelte';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import { client } from '$lib/api/openapi';
	import StopTimes from '$lib/StopTimes.svelte';
	import { onMount, tick } from 'svelte';
	import RailViz from '$lib/RailViz.svelte';
	import MapIcon from 'lucide-svelte/icons/map';
	import { t } from '$lib/i18n/translation';
	import { pushState } from '$app/navigation';
	import { page } from '$app/state';
	import { updateStartDest } from '$lib/updateStartDest';
	import * as Tabs from '$lib/components/ui/tabs';
	import DeparturesMask from '$lib/DeparturesMask.svelte';
	import IsochronesMask from '$lib/IsochronesMask.svelte';
	import Isochrones from '$lib/map/Isochrones.svelte';
	import IsochronesTurf from '$lib/map/IsochronesTurf.svelte';
	// import IsochronesPos from '$lib/map/IsochronesTurf.svelte';

    interface IsochronesPos {
        lat: number;
        lng: number;
        duration: number;
        name?: string;
    };

	const urlParams = browser ? new URLSearchParams(window.location.search) : undefined;
	const hasDebug = urlParams && urlParams.has('debug');
	const hasDark = urlParams && urlParams.has('dark');
	const isSmallScreen = browser && window.innerWidth < 768;
	let dataAttributionLink: string | undefined = $state(undefined);
	let showMap = $state(!isSmallScreen);
	let last_selected_itinerary: Itinerary | undefined = undefined;

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

	onMount(async () => {
		initial().then((d) => {
			if (d.response.headers.has('Link')) {
				dataAttributionLink = d.response.headers
					.get('Link')!
					.replace(/^<(.*)>; rel="license"$/, '$1');
			}
			const r = d.data;
			if (r) {
				center = [r.lon, r.lat];
				zoom = r.zoom;
			}
		});
		await tick();
		applyPageStateFromURL();
	});

	const applyPageStateFromURL = () => {
		if (browser && urlParams) {
			if (urlParams.has('tripId')) {
				onClickTrip(urlParams.get('tripId')!, true);
			}
			if (urlParams.has('stopId')) {
				const time = urlParams.has('time') ? new Date(urlParams.get('time')!) : new Date();
				onClickStop(
					'',
					urlParams.get('stopId')!,
					time,
					urlParams.get('stopArriveBy') == 'true',
					true
				);
			}
		}
	};

	let fromParam: Match | undefined = undefined;
	let toParam: Match | undefined = undefined;
	if (browser && urlParams) {
		fromParam = urlParams.has('from') ? (JSON.parse(urlParams.get('from') ?? '') ?? {}) : undefined;
		toParam = urlParams.has('to') ? (JSON.parse(urlParams.get('to') ?? '') ?? {}) : undefined;
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
	let time = $state<Date>(new Date(urlParams?.get('time') || Date.now()));
	let timeType = $state<string>(urlParams?.get('arriveBy') == 'true' ? 'arrival' : 'departure');
	let wheelchair = $state(urlParams?.get('wheelchair') == 'true');
	let bikeRental = $state(urlParams?.get('bikeRental') == 'true');
	let bikeCarriage = $state(urlParams?.get('bikeCarriage') == 'true');
	let carCarriage = $state(urlParams?.get('carCarriage') == 'true');
	let selectedTransitModes = $state<Mode[] | undefined>(
		(urlParams
			?.get('selectedTransitModes')
			?.split(',')
			.filter((m) => m.length) as Mode[]) ?? undefined
	);
	let firstMileMode = $state<Mode>((urlParams?.get('firstMileMode') ?? 'WALK') as Mode);
	let lastMileMode = $state<Mode>((urlParams?.get('lastMileMode') ?? 'WALK') as Mode);
	let directModes = $state<Mode[]>(
		(urlParams
			?.get('directModes')
			?.split(',')
			.filter((m) => m.length) ?? ['WALK']) as Mode[]
	);
	let elevationCosts = $state<ElevationCosts>(
		(urlParams?.get('elevationCosts') ?? 'NONE') as ElevationCosts
	);
	// let isochronesData = $state<IsochronesPos[]>([
	let isochronesData = $state<IsochronesPos[]>([
		{"lat": 50.767584, "lng": 6.091125, "duration": 15},  // Aachen
		{"lat": 50.73195, "lng": 7.09665, "duration": 5},  // Bonn
		{"lat": 51.517902, "lng": 7.459191, "duration": 10},  // Dortmund
		{"lat": 51.43059, "lng": 6.77599, "duration": 30},  // Duisburg
		{"lat": 51.22005, "lng": 6.79384, "duration": 35},  // Düsseldorf
		{"lat": 50.94293, "lng": 6.95928, "duration": 15},  // Köln
		{"lat": 51.25441, "lng": 7.15013, "duration": 25},  // Wuppertal
	]);

	const toPlaceString = (l: Location) => {
		if (l.value.match?.type === 'STOP') {
			return l.value.match.id;
		} else if (l.value.match?.level) {
			return `${lngLatToStr(l.value.match!)},${l.value.match.level}`;
		} else {
			return `${lngLatToStr(l.value.match!)},0`;
		}
	};
	let additionalModes = $derived([...(bikeRental ? ['RENTAL'] : [])] as Mode[]);
	let preTransitModes = $derived([firstMileMode, ...additionalModes]);
	let postTransitModes = $derived([lastMileMode, ...additionalModes]);
	let requestDirectModes = $derived([...directModes, ...additionalModes]);

	let baseQuery = $derived(
		from.value.match && to.value.match
			? ({
					query: {
						time: time.toISOString(),
						fromPlace: toPlaceString(from),
						toPlace: toPlaceString(to),
						arriveBy: timeType === 'arrival',
						timetableView: true,
						pedestrianProfile: wheelchair ? 'WHEELCHAIR' : 'FOOT',
						preTransitModes,
						postTransitModes,
						directModes: requestDirectModes,
						requireBikeTransport: bikeCarriage,
						requireCarTransport: carCarriage,
						transitModes: selectedTransitModes,
						elevationCosts,
						useRoutedTransfers: true,
						maxMatchingDistance: wheelchair ? 8 : 250
					}
				} as PlanData)
			: undefined
	);

	let searchDebounceTimer: number;
	let baseResponse = $state<Promise<PlanResponse>>();
	let routingResponses = $state<Array<Promise<PlanResponse>>>([]);
	let stopNameFromResponse = $state<string>('');
	$effect(() => {
		if (baseQuery) {
			clearTimeout(searchDebounceTimer);
			searchDebounceTimer = setTimeout(() => {
				const base = plan(baseQuery).then(updateStartDest(from, to));
				baseResponse = base;
				routingResponses = [base];
				pushStateWithQueryString(
					{
						from: JSON.stringify(from?.value?.match),
						to: JSON.stringify(to?.value?.match),
						time,
						arriveBy: timeType === 'arrival',
						wheelchair,
						bikeRental,
						bikeCarriage,
						carCarriage,
						firstMileMode,
						lastMileMode,
						directModes: directModes.join(','),
						elevationCosts,
						selectedTransitModes: selectedTransitModes?.join(',') ?? ''
					},
					{},
					true
				);
			}, 400);
		}
	});

	if (browser) {
		addEventListener('paste', (event) => {
			const paste = event.clipboardData!.getData('text');
			const json = JSON.parse(paste);
			console.log('paste: ', json);
			const response = new Promise<PlanResponse>((resolve, _) => {
				resolve(json as PlanResponse);
			});
			baseResponse = response;
			routingResponses = [response];
		});
	}

	const flyToSelectedItinerary = () => {
		if (last_selected_itinerary === page.state.selectedItinerary) {
			return;
		}
		if (page.state.selectedItinerary && map) {
			const start = maplibregl.LngLat.convert(page.state.selectedItinerary.legs[0].from);
			const box = new maplibregl.LngLatBounds(start, start);
			page.state.selectedItinerary.legs.forEach((l) => {
				box.extend(l.from);
				box.extend(l.to);
				l.intermediateStops?.forEach((x) => {
					box.extend(x);
				});
			});
			const padding = {
				top: isSmallScreen ? Math.max(window.innerHeight / 2, 400) : 96,
				right: 96,
				bottom: 96,
				left: isSmallScreen ? 96 : 640
			};
			map.flyTo({ ...map.cameraForBounds(box, { padding }) });
		}
		last_selected_itinerary = page.state.selectedItinerary;
	};

	$effect(() => {
		flyToSelectedItinerary();
	});

	type CloseFn = () => void;
</script>

{#snippet contextMenu(e: maplibregl.MapMouseEvent, close: CloseFn)}
	<Button
		variant="outline"
		onclick={() => {
			from = posToLocation(e.lngLat, level);
			fromMarker?.setLngLat(from.value.match!);
			close();
		}}
	>
		From
	</Button>
	<Button
		variant="outline"
		onclick={() => {
			to = posToLocation(e.lngLat, level);
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
	bind:center
	transformRequest={(url: string) => {
		if (url.startsWith('/sprite')) {
			return { url: `${window.location.origin}${url}` };
		}
		if (url.startsWith('/')) {
			return { url: `${client.getConfig().baseUrl}/tiles${url}` };
		}
	}}
	class={cn('h-dvh overflow-clip', theme)}
	style={showMap ? getStyle(theme, level) : undefined}
	attribution={false}
>
	{#if hasDebug}
		<Control position="top-right">
			<Debug {bounds} {level} />
		</Control>
	{/if}

	<LevelSelect {bounds} {zoom} bind:level />

	<div class="maplibregl-control-container">
		<div class="maplibregl-ctrl-top-left">
			<Control
				class={isSmallScreen && (page.state.selectedItinerary || page.state.selectedStop)
					? 'hide'
					: ''}
			>
				<Tabs.Root value="connections" class="max-w-full w-[520px] overflow-y-auto">
					<Tabs.List class="grid grid-cols-3">
						<Tabs.Trigger value="connections">{t.connections}</Tabs.Trigger>
						<Tabs.Trigger value="departures">{t.departures}</Tabs.Trigger>
						<Tabs.Trigger value="isochrones">Isochrones</Tabs.Trigger>
					</Tabs.List>
					<Tabs.Content value="connections">
						<Card class="overflow-y-auto overflow-x-hidden bg-background rounded-lg">
							<SearchMask
								geocodingBiasPlace={center}
								bind:from
								bind:to
								bind:time
								bind:timeType
								bind:wheelchair
								bind:bikeRental
								bind:bikeCarriage
								bind:carCarriage
								bind:selectedModes={selectedTransitModes}
								bind:firstMileMode
								bind:lastMileMode
								bind:directModes
								bind:elevationCosts
							/>
						</Card>
					</Tabs.Content>
					<Tabs.Content value="departures">
						<Card class="overflow-y-auto overflow-x-hidden bg-background rounded-lg">
							<DeparturesMask bind:time />
						</Card>
					</Tabs.Content>
					<Tabs.Content value="isochrones">
						<Card class="overflow-y-auto overflow-x-hidden bg-background rounded-lg">
							<IsochronesMask
								geocodingBiasPlace={center}
								bind:isochronesData
								bind:time
							/>
						</Card>
					</Tabs.Content>
				</Tabs.Root>
			</Control>

			{#if routingResponses.length !== 0 && !page.state.showDepartures}
				<Control class="min-h-0 md:mb-2 {page.state.selectedItinerary ? 'hide' : ''}">
					<Card
						class="w-[520px] h-full md:max-h-[70vh] overflow-y-auto overflow-x-hidden bg-background rounded-lg"
					>
						<ItineraryList
							{baseResponse}
							{routingResponses}
							{baseQuery}
							selectItinerary={(selectedItinerary) => pushState('', { selectedItinerary })}
							updateStartDest={updateStartDest(from, to)}
						/>
					</Card>
				</Control>
			{/if}

			{#if page.state.selectedItinerary && !page.state.showDepartures}
				<Control class="min-h-0 mb-12 md:mb-2">
					<Card class="w-[520px] h-full bg-background rounded-lg flex flex-col">
						<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
							<h2 class="ml-2 text-base font-semibold">{t.journeyDetails}</h2>
							<Button
								variant="ghost"
								onclick={() => {
									closeItinerary();
								}}
							>
								<X />
							</Button>
						</div>
						<div
							class={'p-2 md:p-4 overflow-y-auto overflow-x-hidden min-h-0 ' +
								(showMap ? 'max-h-[40vh] md:max-h-[70vh]' : '')}
						>
							<ConnectionDetail itinerary={page.state.selectedItinerary} />
						</div>
					</Card>
				</Control>
				{#if showMap}
					<ItineraryGeoJson itinerary={page.state.selectedItinerary} {level} />
				{/if}
			{/if}

			{#if page.state.selectedStop && page.state.showDepartures}
				<Control class="min-h-0 md:mb-2">
					<Card class="w-[520px] h-full bg-background rounded-lg flex flex-col">
						<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
							<h2 class="ml-2 text-base font-semibold">
								{#if page.state.stopArriveBy}
									{t.arrivals}
								{:else}
									{t.departures}
								{/if}
								in
								{stopNameFromResponse}
							</h2>
							<Button
								variant="ghost"
								onclick={() => {
									pushStateWithQueryString(
										{ ...(page.state.tripId && { tripId: page.state.tripId }) },
										{ selectedItinerary: page.state.selectedItinerary }
									);
								}}
							>
								<X />
							</Button>
						</div>
						<div class="p-2 md:p-4 overflow-y-auto overflow-x-hidden min-h-0 md:max-h-[70vh]">
							<StopTimes
								stopId={page.state.selectedStop.stopId}
								stopName={page.state.selectedStop.name}
								time={page.state.selectedStop.time}
								bind:stopNameFromResponse
								arriveBy={page.state.stopArriveBy}
							/>
						</div>
					</Card>
				</Control>
			{/if}
		</div>
	</div>

	<div class="maplibregl-ctrl-bottom-right">
		<div class="maplibregl-ctrl maplibregl-ctrl-attrib">
			<div class="maplibregl-ctrl-attrib-inner">
				&copy; <a href="http://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>
				{#if dataAttributionLink}
					| <a href={dataAttributionLink} target="_blank">{t.timetableSources}</a>
				{/if}
			</div>
		</div>
	</div>

	{#if showMap}
		<RailViz {map} {bounds} {zoom} />
		<!-- <Isochrones class="absolute flex border-2 border-indigo-600 h-dvh max-w-full z-50" {bounds} {isochronesData} /> -->
		<IsochronesTurf {map} {bounds} {isochronesData} />

		<Popup trigger="contextmenu" children={contextMenu} />

		{#if from}
			<Marker
				color="green"
				draggable={true}
				{level}
				bind:location={from}
				bind:marker={fromMarker}
			/>
		{/if}

		{#if to}
			<Marker color="red" draggable={true} {level} bind:location={to} bind:marker={toMarker} />
		{/if}
	{:else}
		<div class="maplibregl-control-container">
			<div class="maplibregl-ctrl-bottom-left">
				<Control class="pb-4">
					<Button
						size="icon"
						variant="default"
						onclick={() => {
							showMap = true;
							flyToSelectedItinerary();
						}}
					>
						<MapIcon class="h-[1.2rem] w-[1.2rem]" />
					</Button>
				</Control>
			</div>
		</div>
	{/if}
</Map>
