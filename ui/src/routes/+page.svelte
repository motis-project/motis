<script lang="ts">
	import { X, Palette, Rss, Ban, LocateFixed, TrainFront } from '@lucide/svelte';
	import { getStyle } from '$lib/map/style';
	import Map from '$lib/map/Map.svelte';
	import Control from '$lib/map/Control.svelte';
	import SearchMask from '$lib/SearchMask.svelte';
	import { parseLocation, posToLocation, type Location } from '$lib/Location';
	import { Card } from '$lib/components/ui/card';
	import {
		initial,
		oneToAll,
		plan,
		type ElevationCosts,
		type OneToAllData,
		type OneToAllResponse,
		type PlanResponse,
		type Itinerary,
		type Mode,
		type PedestrianProfile,
		type PlanData,
		type ReachablePlace,
		type RentalFormFactor,
		type ServerConfig
	} from '@motis-project/motis-client';
	import ItineraryList from '$lib/ItineraryList.svelte';
	import ConnectionDetail from '$lib/ConnectionDetail.svelte';
	import { Button } from '$lib/components/ui/button';
	import ItineraryGeoJson from '$lib/map/itineraries/ItineraryGeoJSON.svelte';
	import maplibregl from 'maplibre-gl';
	import { browser } from '$app/environment';
	import { cn, getUrlArray, onClickStop, onClickTrip, pushStateWithQueryString } from '$lib/utils';
	import Debug from '$lib/Debug.svelte';
	import Marker from '$lib/map/Marker.svelte';
	import Popup from '$lib/map/Popup.svelte';
	import LevelSelect from '$lib/LevelSelect.svelte';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import Drawer from '$lib/map/Drawer.svelte';
	import { client } from '@motis-project/motis-client';
	import StopTimes from '$lib/StopTimes.svelte';
	import { onMount, tick, untrack } from 'svelte';
	import RailViz from '$lib/RailViz.svelte';
	import { t } from '$lib/i18n/translation';
	import { pushState } from '$app/navigation';
	import { page } from '$app/state';
	import { preprocessItinerary } from '$lib/preprocessItinerary';
	import * as Tabs from '$lib/components/ui/tabs';
	import DeparturesMask from '$lib/DeparturesMask.svelte';
	import Isochrones from '$lib/map/Isochrones.svelte';
	import IsochronesInfo from '$lib/IsochronesInfo.svelte';
	import type { DisplayLevel, IsochronesOptions, IsochronesPos } from '$lib/map/IsochronesShared';
	import IsochronesMask from '$lib/IsochronesMask.svelte';
	import Rentals from '$lib/map/rentals/Rentals.svelte';
	import {
		getFormFactors,
		getPrePostDirectModes,
		possibleTransitModes,
		prePostModesToModes,
		type PrePostDirectMode
	} from '$lib/Modes';
	import { defaultQuery, omitDefaults } from '$lib/defaults';
	import { LEVEL_MIN_ZOOM } from '$lib/constants';
	import StopGeoJSON from '$lib/map/stops/StopsGeoJSON.svelte';

	const urlParams = browser ? new URLSearchParams(window.location.search) : undefined;

	const hasDebug: boolean = Boolean(urlParams?.has('debug'));
	const hasDark: boolean = Boolean(urlParams?.has('dark'));
	const hasLight: boolean = Boolean(urlParams?.has('light'));
	const isSmallScreen = browser && window.innerWidth < 768;
	let activeTab = $state<'connections' | 'departures' | 'isochrones'>('connections');
	let dataAttributionLink: string | undefined = $state(undefined);
	let colorMode = $state<'rt' | 'route' | 'mode' | 'none'>('none');
	let showMap = $state(!isSmallScreen);
	let lastSelectedItinerary: Itinerary | undefined = undefined;
	let lastOneToAllQuery: OneToAllData | undefined = undefined;
	let serverConfig: ServerConfig | undefined = $state();

	$effect(() => {
		if (activeTab == 'isochrones') {
			colorMode = 'none';
		}
	});

	let theme: 'light' | 'dark' =
		(hasDark ? 'dark' : hasLight ? 'light' : undefined) ??
		(browser && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
			? 'dark'
			: 'light');
	if (theme === 'dark') {
		document.documentElement.classList.add('dark');
	}

	let center = $state.raw<[number, number]>([2.258882912876089, 48.72559118651327]);
	let level = $state(0);
	let zoom = $state(15);
	let bounds = $state<maplibregl.LngLatBoundsLike>();
	let map = $state<maplibregl.Map>();

	const geolocate = new maplibregl.GeolocateControl({
		positionOptions: {
			enableHighAccuracy: true
		},
		showAccuracyCircle: false
	});

	const getLocation = () => {
		geolocate.trigger();
	};

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
				serverConfig = r.serverConfig;
			}
		});
		await tick();
		applyPageStateFromURL();
	});

	const applyPageStateFromURL = () => {
		if (browser && urlParams) {
			const tripId = urlParams.get('tripId');
			if (tripId !== null) {
				onClickTrip(tripId, true);
			}

			const stopId = urlParams.get('stopId');
			if (stopId !== null) {
				const time = urlParams.has('time') ? new Date(urlParams.get('time')!) : new Date();
				onClickStop('', stopId, time, urlParams.get('stopArriveBy') == 'true', true);
			}
		}
	};

	function parseIntOr<T>(s: string | null | undefined, d: T): T | number {
		if (s) {
			const v = parseInt(s);
			return isNaN(v) ? d : v;
		} else {
			return d;
		}
	}

	let fromMarker = $state<maplibregl.Marker>();
	let toMarker = $state<maplibregl.Marker>();
	let oneMarker = $state<maplibregl.Marker>();
	let stopMarker = $state<maplibregl.Marker>();
	let from = $state<Location>(
		parseLocation(urlParams?.get('fromPlace'), urlParams?.get('fromName'))
	);
	let to = $state<Location>(parseLocation(urlParams?.get('toPlace'), urlParams?.get('toName')));
	let one = $state<Location>(parseLocation(urlParams?.get('one'), urlParams?.get('oneName')));
	let stop = $state<Location>();
	let time = $state<Date>(new Date(urlParams?.get('time') || Date.now()));
	let timetableView = $state(urlParams?.get('timetableView') != 'false');
	let searchWindow = $state(
		urlParams?.get('searchWindow')
			? parseInt(urlParams.get('searchWindow')!)
			: defaultQuery.searchWindow
	);
	let numItineraries = $state(
		urlParams?.get('numItineraries')
			? parseIntOr(urlParams.get('numItineraries'), defaultQuery.numItineraries)
			: defaultQuery.numItineraries
	);
	let maxItineraries = $state(
		urlParams?.get('maxItineraries')
			? parseIntOr(urlParams.get('maxItineraries'), undefined)
			: undefined
	);
	let arriveBy = $state<boolean>(urlParams?.get('arriveBy') == 'true');
	let algorithm = $state<PlanData['query']['algorithm']>(
		(urlParams?.get('algorithm') ?? defaultQuery.algorithm) as PlanData['query']['algorithm']
	);
	let useRoutedTransfers = $state(
		urlParams?.get('useRoutedTransfers') == 'true' || defaultQuery.useRoutedTransfers
	);
	let pedestrianProfile = $state<PedestrianProfile>(
		(urlParams?.has('pedestrianProfile')
			? urlParams.get('pedestrianProfile')
			: defaultQuery.pedestrianProfile) as PedestrianProfile
	);
	let requireBikeTransport = $state(urlParams?.get('requireBikeTransport') == 'true');
	let requireCarTransport = $state(urlParams?.get('requireCarTransport') == 'true');
	let transitModes = $state<Mode[]>(
		getUrlArray('transitModes', defaultQuery.transitModes) as Mode[]
	);
	let preTransitModes = $state<PrePostDirectMode[]>(
		getPrePostDirectModes(
			getUrlArray('preTransitModes', defaultQuery.preTransitModes) as Mode[],
			getUrlArray('preTransitRentalFormFactors') as RentalFormFactor[]
		)
	);
	let postTransitModes = $state<PrePostDirectMode[]>(
		getPrePostDirectModes(
			getUrlArray('postTransitModes', defaultQuery.postTransitModes) as Mode[],
			getUrlArray('postTransitRentalFormFactors') as RentalFormFactor[]
		)
	);
	let directModes = $state<PrePostDirectMode[]>(
		getPrePostDirectModes(
			getUrlArray('directModes', defaultQuery.directModes) as Mode[],
			getUrlArray('directRentalFormFactors') as RentalFormFactor[]
		)
	);
	let preTransitProviderGroups = $state<string[]>(getUrlArray('preTransitRentalProviderGroups'));
	let postTransitProviderGroups = $state<string[]>(getUrlArray('postTransitRentalProviderGroups'));
	let directProviderGroups = $state<string[]>(getUrlArray('directRentalProviderGroups'));
	let elevationCosts = $state<ElevationCosts>(
		(urlParams?.get('elevationCosts') ?? 'NONE') as ElevationCosts
	);
	let maxTransfers = $state<number>(
		parseIntOr(urlParams?.get('maxTransfers'), defaultQuery.maxTransfers)
	);
	let maxTravelTime = $derived<number>(
		parseIntOr(
			urlParams?.get('maxTravelTime'),
			Math.min(
				defaultQuery.maxTravelTime,
				60 * (serverConfig?.maxOneToAllTravelTimeLimit ?? Infinity)
			)
		)
	);
	let maxPreTransitTime = $derived<number>(
		parseIntOr(
			urlParams?.get('maxPreTransitTime'),
			Math.min(defaultQuery.maxPreTransitTime, serverConfig?.maxPrePostTransitTimeLimit ?? Infinity)
		)
	);
	let maxPostTransitTime = $derived<number>(
		parseIntOr(
			urlParams?.get('maxPostTransitTime'),
			Math.min(
				defaultQuery.maxPostTransitTime,
				serverConfig?.maxPrePostTransitTimeLimit ?? Infinity
			)
		)
	);
	let maxDirectTime = $derived<number>(
		parseIntOr(
			urlParams?.get('maxDirectTime'),
			Math.min(defaultQuery.maxDirectTime, serverConfig?.maxDirectTimeLimit ?? Infinity)
		)
	);
	let ignorePreTransitRentalReturnConstraints = $state(
		urlParams?.get('ignorePreTransitRentalReturnConstraints') == 'true'
	);
	let ignorePostTransitRentalReturnConstraints = $state(
		urlParams?.get('ignorePostTransitRentalReturnConstraints') == 'true'
	);
	let ignoreDirectRentalReturnConstraints = $state(
		urlParams?.get('ignoreDirectRentalReturnConstraints') == 'true'
	);
	let slowDirect = $state(urlParams?.get('slowDirect') == 'true');

	let isochronesData = $state<IsochronesPos[]>([]);
	let isochronesOptions = $state<IsochronesOptions>({
		displayLevel:
			(urlParams?.get('isochronesDisplayLevel') as DisplayLevel) ??
			defaultQuery.isochronesDisplayLevel,
		color: urlParams?.get('isochronesColor') ?? defaultQuery.isochronesColor,
		opacity: parseIntOr(urlParams?.get('isochronesOpacity'), defaultQuery.isochronesOpacity),
		status: 'DONE',
		error: undefined
	});
	const isochronesCircleResolution = urlParams?.get('isochronesCircleResolution')
		? parseIntOr(urlParams.get('isochronesCircleResolution'), defaultQuery.circleResolution)
		: defaultQuery.circleResolution;

	const toPlaceString = (l: Location) => {
		if (l.match?.type === 'STOP') {
			return l.match.id;
		} else if (l.match?.level) {
			return `${lngLatToStr(l.match!)},${l.match.level}`;
		} else {
			return `${lngLatToStr(l.match!)}`;
		}
	};

	const providerGroupsForQuery = (modes: PrePostDirectMode[], groups: string[]): string[] => {
		if (!modes.some((mode) => mode.startsWith('RENTAL_'))) {
			return [];
		}
		return Array.from(new Set(groups));
	};

	let baseQuery = $derived(
		from.match && to.match
			? ({
					query: omitDefaults({
						time: time.toISOString(),
						fromPlace: toPlaceString(from),
						toPlace: toPlaceString(to),
						arriveBy,
						timetableView,
						searchWindow,
						numItineraries,
						maxItineraries,
						withFares: true,
						slowDirect,
						fastestDirectFactor: 1.5,
						pedestrianProfile,
						joinInterlinedLegs: false,
						transitModes:
							transitModes.length == possibleTransitModes.length
								? defaultQuery.transitModes
								: transitModes,
						preTransitModes: prePostModesToModes(preTransitModes),
						postTransitModes: prePostModesToModes(postTransitModes),
						directModes: prePostModesToModes(directModes),
						preTransitRentalFormFactors: getFormFactors(preTransitModes),
						postTransitRentalFormFactors: getFormFactors(postTransitModes),
						directRentalFormFactors: getFormFactors(directModes),
						preTransitRentalProviderGroups: providerGroupsForQuery(
							preTransitModes,
							preTransitProviderGroups
						),
						postTransitRentalProviderGroups: providerGroupsForQuery(
							postTransitModes,
							postTransitProviderGroups
						),
						directRentalProviderGroups: providerGroupsForQuery(directModes, directProviderGroups),
						requireBikeTransport,
						requireCarTransport,
						elevationCosts,
						useRoutedTransfers,
						maxTransfers: maxTransfers,
						maxMatchingDistance: pedestrianProfile == 'WHEELCHAIR' ? 8 : 250,
						maxPreTransitTime,
						maxPostTransitTime,
						maxDirectTime,
						ignorePreTransitRentalReturnConstraints,
						ignorePostTransitRentalReturnConstraints,
						ignoreDirectRentalReturnConstraints,
						algorithm
					} as PlanData['query'])
				} as PlanData)
			: undefined
	);
	let isochronesQuery = $derived(
		one?.match
			? ({
					query: {
						one: toPlaceString(one),
						maxTravelTime: Math.ceil(maxTravelTime / 60),
						time: time.toISOString(),
						transitModes,
						maxTransfers,
						arriveBy,
						useRoutedTransfers,
						pedestrianProfile,
						requireBikeTransport,
						requireCarTransport,
						preTransitModes: prePostModesToModes(preTransitModes),
						postTransitModes: prePostModesToModes(postTransitModes),
						maxPreTransitTime,
						maxPostTransitTime,
						elevationCosts,
						maxMatchingDistance: pedestrianProfile == 'WHEELCHAIR' ? 8 : 250,
						ignorePreTransitRentalReturnConstraints,
						ignorePostTransitRentalReturnConstraints
					}
				} as OneToAllData)
			: undefined
	);

	let searchDebounceTimer: number;
	let baseResponse = $state<Promise<PlanResponse>>();
	let routingResponses = $state<Array<Promise<PlanResponse>>>([]);
	let stopNameFromResponse = $state<string>('');
	$effect(() => {
		if (baseQuery && activeTab == 'connections') {
			clearTimeout(searchDebounceTimer);
			searchDebounceTimer = setTimeout(() => {
				const base = plan(baseQuery).then(preprocessItinerary(from, to));
				const q = baseQuery.query;
				baseResponse = base;
				routingResponses = [base];
				pushStateWithQueryString(
					{
						...q,
						...(q.fromPlace == from.label ? {} : { fromName: from.label }),
						...(q.toPlace == to.label ? {} : { toName: to.label })
					},
					{},
					true
				);
			}, 400);
		}
	});
	let isochronesQueryTimeout: number;
	$effect(() => {
		if (isochronesQuery && activeTab == 'isochrones') {
			const [isochronesColor, isochronesOpacity, isochronesDisplayLevel] = [
				isochronesOptions.color,
				isochronesOptions.opacity,
				isochronesOptions.displayLevel
			];
			if (lastOneToAllQuery != isochronesQuery) {
				lastOneToAllQuery = isochronesQuery;
				clearTimeout(isochronesQueryTimeout);
				isochronesOptions.status = 'WORKING';
				isochronesOptions.error = undefined;
				isochronesQueryTimeout = setTimeout(() => {
					oneToAll(isochronesQuery)
						.then((r: { data: OneToAllResponse | undefined; error: unknown }) => {
							if (r.error) {
								const msg = (r.error as { error: string }).error;
								throw new Error(String(msg));
							}
							const all = r.data!.all!.map((p: ReachablePlace) => {
								return {
									lat: p.place?.lat,
									lng: p.place?.lon,
									seconds: maxTravelTime - 60 * (p.duration ?? 0),
									name: p.place?.name
								} as IsochronesPos;
							});
							isochronesData = [...all];
							isochronesOptions.status = isochronesData.length == 0 ? 'EMPTY' : 'WORKING';
						})
						.catch((e: Error) => {
							isochronesOptions.status = 'FAILED';
							isochronesOptions.error = e.message;
						});
				}, 60);
			}
			untrack(() => {
				const q = isochronesQuery.query;
				pushStateWithQueryString(
					{
						...q,
						...(q.one == one.label ? {} : { oneName: one.label }),
						maxTravelTime: q.maxTravelTime * 60,
						isochronesColor,
						isochronesOpacity,
						isochronesDisplayLevel,
						...(isochronesCircleResolution && isochronesCircleResolution > 2
							? { isochronesCircleResolution }
							: {})
					},
					{},
					true
				);
			});
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

	const flyToItineraries = (itineraries: Itinerary[], map: maplibregl.Map) => {
		const start = maplibregl.LngLat.convert(itineraries[0].legs[0].from);
		const box = new maplibregl.LngLatBounds(start, start);
		itineraries.forEach((i) => {
			i.legs.forEach((l) => {
				box.extend(l.from);
				box.extend(l.to);
				l.intermediateStops?.forEach((x) => {
					box.extend(x);
				});
			});
		});
		const padding = {
			top: 96,
			right: 96,
			bottom: isSmallScreen ? window.innerHeight * 0.3 : 96,
			left: isSmallScreen ? 96 : 640
		};
		map.flyTo({ ...map.cameraForBounds(box, { padding }) });
	};

	const flyToSelectedItinerary = () => {
		if (lastSelectedItinerary === page.state.selectedItinerary) {
			return;
		}
		if (page.state.selectedItinerary && map) {
			flyToItineraries([page.state.selectedItinerary], map);
		}
		lastSelectedItinerary = page.state.selectedItinerary;
	};

	$effect(() => {
		if (map) {
			map.addControl(geolocate);
		}
	});

	$effect(flyToSelectedItinerary);

	$effect(() => {
		Promise.all(routingResponses).then((responses) => {
			if (map) {
				flyToItineraries(
					responses.flatMap((response) => response.itineraries),
					map
				);
			}
		});
	});

	type CloseFn = () => void;
</script>

{#snippet contextMenu(e: maplibregl.MapMouseEvent, close: CloseFn)}
	{#if activeTab == 'connections'}
		<Button
			variant="outline"
			onclick={() => {
				from = posToLocation(e.lngLat, zoom > LEVEL_MIN_ZOOM ? level : undefined);
				fromMarker?.setLngLat(from.match!);
				close();
			}}
		>
			From
		</Button>
		<Button
			variant="outline"
			onclick={() => {
				to = posToLocation(e.lngLat, zoom > LEVEL_MIN_ZOOM ? level : undefined);
				toMarker?.setLngLat(to.match!);
				close();
			}}
		>
			To
		</Button>
	{:else if activeTab == 'isochrones'}
		<Button
			variant="outline"
			onclick={() => {
				one = posToLocation(e.lngLat, zoom > LEVEL_MIN_ZOOM ? level : undefined);
				oneMarker?.setLngLat(one.match!);
				close();
			}}
		>
			{t.position}
		</Button>
	{/if}
{/snippet}

{#snippet resultContent()}
	<Control>
		<Tabs.Root bind:value={activeTab} class="max-w-full w-[520px] overflow-y-auto">
			<Tabs.List class="grid grid-cols-3">
				<Tabs.Trigger value="connections">{t.connections}</Tabs.Trigger>
				<Tabs.Trigger value="departures">{t.departures}</Tabs.Trigger>
				<Tabs.Trigger value="isochrones">{t.isochrones.title}</Tabs.Trigger>
			</Tabs.List>
			<Tabs.Content value="connections">
				<Card class="overflow-y-auto overflow-x-hidden bg-background rounded-lg">
					<SearchMask
						geocodingBiasPlace={center}
						{serverConfig}
						bind:from
						bind:to
						bind:time
						bind:arriveBy
						bind:useRoutedTransfers
						bind:maxTransfers
						bind:pedestrianProfile
						bind:requireCarTransport
						bind:requireBikeTransport
						bind:transitModes
						bind:preTransitModes
						bind:postTransitModes
						bind:directModes
						bind:elevationCosts
						bind:maxPreTransitTime
						bind:maxPostTransitTime
						bind:maxDirectTime
						bind:ignorePreTransitRentalReturnConstraints
						bind:ignorePostTransitRentalReturnConstraints
						bind:ignoreDirectRentalReturnConstraints
						bind:preTransitProviderGroups
						bind:postTransitProviderGroups
						bind:directProviderGroups
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
						bind:one
						{serverConfig}
						bind:maxTravelTime
						geocodingBiasPlace={center}
						bind:time
						bind:useRoutedTransfers
						bind:pedestrianProfile
						bind:requireCarTransport
						bind:requireBikeTransport
						bind:transitModes
						bind:maxTransfers
						bind:preTransitModes
						bind:postTransitModes
						bind:maxPreTransitTime
						bind:maxPostTransitTime
						bind:arriveBy
						bind:elevationCosts
						bind:ignorePreTransitRentalReturnConstraints
						bind:ignorePostTransitRentalReturnConstraints
						bind:options={isochronesOptions}
						bind:preTransitProviderGroups
						bind:postTransitProviderGroups
						bind:directProviderGroups
					/>
				</Card>
			</Tabs.Content>
		</Tabs.Root>
	</Control>

	{#if activeTab == 'connections' && routingResponses.length !== 0 && !page.state.showDepartures}
		<Control class="min-h-0 md:mb-2 {page.state.selectedItinerary ? 'hide' : ''}">
			<Card
				class="scrollable w-[520px] h-full md:max-h-[60vh] {isSmallScreen
					? 'border-0 shadow-none'
					: ''} overflow-x-hidden bg-background rounded-lg mb-2"
			>
				<ItineraryList
					{baseResponse}
					{routingResponses}
					{baseQuery}
					selectItinerary={(selectedItinerary) => {
						pushState('', { selectedItinerary: selectedItinerary, scrollY: undefined });
					}}
					updateStartDest={preprocessItinerary(from, to)}
				/>
			</Card>
		</Control>
		{#if showMap && !page.state.selectedItinerary}
			{#each routingResponses as r, rI (rI)}
				{#await r then r}
					{#each r.itineraries as it, i (i)}
						<ItineraryGeoJson
							itinerary={it}
							id="{rI}-{i}"
							selected={false}
							selectItinerary={() => {
								pushState('', { selectedItinerary: it });
							}}
							{level}
							{theme}
						/>
					{/each}
				{/await}
			{/each}
		{/if}
	{/if}

	{#if activeTab != 'isochrones' && page.state.selectedItinerary && !page.state.showDepartures}
		<Control class="min-h-0 md:mb-2">
			<Card class="w-[520px] md:max-h-[60vh] h-full bg-background rounded-lg flex flex-col mb-2">
				<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
					<h2 class="ml-2 text-base font-semibold">{t.journeyDetails}</h2>
					<Button
						variant="ghost"
						onclick={() => {
							history.back();
						}}
					>
						<X />
					</Button>
				</div>
				<div
					class={'p-2 md:p-4 overflow-y-auto overflow-x-hidden min-h-0 ' +
						(showMap ? 'md:max-h-[60vh]' : '')}
				>
					<ConnectionDetail itinerary={page.state.selectedItinerary} />
				</div>
			</Card>
		</Control>
		{#if showMap}
			<ItineraryGeoJson itinerary={page.state.selectedItinerary} selected={true} {level} {theme} />
			<StopGeoJSON itinerary={page.state.selectedItinerary} {theme} />
		{/if}
	{/if}

	{#if activeTab != 'isochrones' && page.state.selectedStop && page.state.showDepartures}
		<Control class="min-h-0 md:mb-2">
			<Card class="w-[520px] md:max-h-[60vh] h-full bg-background rounded-lg flex flex-col mb-2">
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
							history.back();
						}}
					>
						<X />
					</Button>
				</div>
				<div class="p-2 md:p-4 overflow-y-auto overflow-x-hidden min-h-0 md:max-h-[60vh]">
					<StopTimes
						stopId={page.state.selectedStop.stopId}
						stopName={page.state.selectedStop.name}
						time={page.state.selectedStop.time}
						bind:stop
						bind:stopMarker
						bind:stopNameFromResponse
						arriveBy={page.state.stopArriveBy}
					/>
				</div>
			</Card>
		</Control>
	{/if}

	{#if activeTab == 'isochrones' && one.match}
		<Control class="min-h-0 md:mb-2 {isochronesOptions.status == 'DONE' ? 'hide' : ''}">
			<Card class="w-[520px] overflow-y-auto overflow-x-hidden bg-background rounded-lg">
				<IsochronesInfo options={isochronesOptions} />
			</Card>
		</Control>
	{/if}
{/snippet}

<Map
	bind:map
	bind:bounds
	bind:zoom
	bind:center
	class={cn('h-dvh overflow-clip', theme)}
	style={showMap && browser
		? getStyle(
				theme,
				level,
				window.location.origin + window.location.pathname,
				client.getConfig().baseUrl
					? client.getConfig().baseUrl + '/'
					: window.location.origin + window.location.pathname
			)
		: undefined}
	attribution={false}
>
	{#if hasDebug}
		<Control position="top-right" class="text-right">
			<Debug {bounds} {level} {zoom} />
		</Control>
	{/if}

	<LevelSelect {bounds} {zoom} bind:level />

	{#if browser}
		{#if isSmallScreen}
			<Drawer class="relative z-10 h-full mt-5 flex flex-col" bind:showMap>
				{@render resultContent()}
			</Drawer>
		{:else}
			<div class="maplibregl-ctrl-top-left">
				{@render resultContent()}
			</div>
		{/if}
	{/if}

	<div class="maplibregl-ctrl-{isSmallScreen ? 'top-left' : 'bottom-right'}">
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
		{#if activeTab != 'isochrones'}
			<Control position="top-right" class="pb-4 text-right">
				<Button
					size="icon"
					onclick={() => {
						colorMode = (function () {
							switch (colorMode) {
								case 'rt':
									return 'route';
								case 'route':
									return 'mode';
								case 'mode':
									return 'none';
								case 'none':
									return 'rt';
							}
						})();
					}}
				>
					{#if colorMode == 'rt'}
						<Rss class="h-[1.2rem] w-[1.2rem]" />
					{:else if colorMode == 'mode'}
						<TrainFront class="h-[1.2rem] w-[1.2rem]" />
					{:else if colorMode == 'none'}
						<Ban class="h-[1.2rem] w-[1.2rem]" />
					{:else}
						<Palette class="h-[1.2rem] w-[1.2rem]" />
					{/if}
				</Button>
				<Button size="icon" onclick={() => getLocation()}>
					<LocateFixed class="w-5 h-5" />
				</Button>
			</Control>
			<Rentals {map} {bounds} {zoom} {theme} debug={hasDebug} />
		{/if}

		<RailViz {map} {bounds} {zoom} {colorMode} />
		<Isochrones
			{map}
			{bounds}
			{isochronesData}
			streetModes={arriveBy ? preTransitModes : postTransitModes}
			wheelchair={pedestrianProfile === 'WHEELCHAIR'}
			maxAllTime={arriveBy ? maxPreTransitTime : maxPostTransitTime}
			circleResolution={isochronesCircleResolution}
			active={activeTab == 'isochrones'}
			bind:options={isochronesOptions}
		/>

		<Popup trigger="contextmenu" children={contextMenu} />

		{#if from && activeTab == 'connections'}
			<Marker
				color="green"
				draggable={true}
				{level}
				bind:location={from}
				bind:marker={fromMarker}
			/>
		{/if}

		{#if stop && page.state.showDepartures && activeTab != 'isochrones'}
			<Marker
				color="black"
				draggable={false}
				{level}
				bind:location={stop}
				bind:marker={stopMarker}
			/>
		{/if}

		{#if to && activeTab == 'connections'}
			<Marker color="red" draggable={true} {level} bind:location={to} bind:marker={toMarker} />
		{/if}

		{#if one && activeTab == 'isochrones'}
			<Marker color="yellow" draggable={true} {level} bind:location={one} bind:marker={oneMarker} />
		{/if}
	{/if}
</Map>
