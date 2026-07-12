<script lang="ts">
	import {
		X,
		Palette,
		Rss,
		Ban,
		LocateFixed,
		MapPin,
		TrainFront,
		Waypoints,
		MountainSnow,
		Compass,
		RefreshCw
	} from '@lucide/svelte';
	import { MediaQuery } from 'svelte/reactivity';
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
		type PlanResponse,
		type Itinerary,
		type Mode,
		type PedestrianProfile,
		type Place,
		type PlanData,
		type ReachablePlace,
		type RentalFormFactor,
		type ServerConfig,
		type CyclingSpeed,
		type PedestrianSpeed,
		refreshItinerary,
		type Match
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
	import { language, t } from '$lib/i18n/translation';
	import { pushState, replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import { preprocessItinerary, updateItinerary } from '$lib/preprocessItinerary';
	import * as Tabs from '$lib/components/ui/tabs';
	import * as Select from '$lib/components/ui/select';
	import DeparturesMask from '$lib/DeparturesMask.svelte';
	import Isochrones from '$lib/map/Isochrones.svelte';
	import IsochronesInfo from '$lib/IsochronesInfo.svelte';
	import type { DisplayLevel, IsochronesOptions, IsochronesPos } from '$lib/map/IsochronesShared';
	import IsochronesMask from '$lib/IsochronesMask.svelte';
	import Rentals from '$lib/map/rentals/Rentals.svelte';
	import Routes from '$lib/map/routes/Routes.svelte';
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
	import RailViz from '$lib/RailViz.svelte';
	import StopsView from '$lib/map/stops/StopsView.svelte';
	import { formatDate } from '$lib/toDateTime';
	import { getPageTitle } from '$lib/pageTitle';

	const urlParams = browser ? new URLSearchParams(window.location.search) : undefined;

	const hasDebug: boolean = Boolean(urlParams?.has('debug'));
	const hasDark: boolean = Boolean(urlParams?.has('dark'));
	const hasLight: boolean = Boolean(urlParams?.has('light'));
	const isSmallScreen = new MediaQuery('(max-width: 768px)');
	let activeTab = $derived<'connections' | 'departures' | 'isochrones'>(
		page.state.activeTab ??
			(urlParams?.has('one')
				? 'isochrones'
				: urlParams?.has('stopId')
					? 'departures'
					: 'connections')
	);
	let dataAttributionLink: string | undefined = $state(undefined);
	type ColorMode = 'none' | 'stops' | 'rt' | 'route' | 'mode';
	let colorMode = $state<ColorMode>('stops');
	const colorModeOptions: { value: ColorMode; label: string; icon: typeof Ban }[] = [
		{ value: 'none', label: t.colorMode.none, icon: Ban },
		{ value: 'stops', label: t.colorMode.stops, icon: MapPin },
		{ value: 'route', label: t.colorMode.route, icon: Palette },
		{ value: 'mode', label: t.colorMode.mode, icon: TrainFront },
		{ value: 'rt', label: t.colorMode.rt, icon: Rss }
	];
	let showMap = $state(!isSmallScreen.current);
	let showRoutes = $state(false);
	let lastOneToAllQuery: Parameters<typeof oneToAll>[0] | undefined = undefined;
	let lastPlanQuery: PlanData | undefined = undefined;
	let serverConfig: ServerConfig | undefined = $state();
	let dataLoaded: boolean = $state(false);
	$effect(() => {
		if (!isSmallScreen.current) {
			showMap = true;
		}
	});
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

	let withHillshades = $state(false);
	let center = $state.raw<[number, number]>([2.258882912876089, 48.72559118651327]);
	let level = $state(0);
	let zoom = $state(15);
	let bounds = $state<maplibregl.LngLatBoundsLike>();
	let bearing = $state(0);
	let map = $state<maplibregl.Map>();
	let style = $derived(
		browser
			? getStyle(
					theme,
					level,
					window.location.origin + window.location.pathname,
					client.getConfig().baseUrl
						? client.getConfig().baseUrl + '/'
						: window.location.origin + window.location.pathname,
					withHillshades
				)
			: undefined
	);

	const geolocate = new maplibregl.GeolocateControl({
		positionOptions: {
			enableHighAccuracy: true
		},
		showAccuracyCircle: false,
		trackUserLocation: true
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
			dataLoaded = true;
		});
		await tick();
		applyPageStateFromURL();
	});

	// Drops undefined values so they don't end up as the string "undefined" in the URL.
	const definedOnly = <T extends Record<string, unknown>>(obj: T): T =>
		Object.fromEntries(Object.entries(obj).filter(([, v]) => v !== undefined)) as T;

	// Navigate to the connection detail of an itinerary. Reflects the itinerary ID
	// and all parameters used to reconstruct (refresh) it in the URL query string
	// so that the view can be restored / shared via the URL.
	const onSelectItinerary = (itinerary: Itinerary, replace: boolean = false) => {
		pushStateWithQueryString(
			itinerary.id
				? definedOnly({
						itineraryId: itinerary.id,
						fromName: from.label || undefined,
						toName: to.label || undefined,
						joinInterlinedLegs: false,
						detailedLegs: true,
						detailedTransfers: true,
						withFares: true,
						numLegAlternatives: 3,
						language: [language],
						...refreshLegAlternativeParams
					})
				: {},
			{
				selectedItinerary: itinerary,
				scrollY: undefined,
				selectedStop: replace ? undefined : page.state.selectedStop,
				tripId: replace ? undefined : page.state.tripId,
				activeTab: 'connections'
			},
			replace
		);
	};

	// Reconstruct an itinerary from an itinerary ID using the refresh parameters
	// found in the current URL, then show it in the connection detail view.
	const onShowItineraryId = async (itineraryId: string, replace: boolean = false) => {
		const boolParam = (key: string): boolean | undefined => {
			const v = urlParams?.get(key);
			return v == null ? undefined : v === 'true'; // absent = undefined
		};
		const arrParam = (key: string): string[] | undefined => {
			const a = getUrlArray(key);
			return a.length ? a : undefined; // absent = undefined
		};
		const transitModesUrl = getUrlArray('transitModes');
		const query = definedOnly({
			itineraryId,
			requireDisplayNameMatch: boolParam('requireDisplayNameMatch'),
			joinInterlinedLegs: boolParam('joinInterlinedLegs'),
			detailedTransfers: boolParam('detailedTransfers'),
			detailedLegs: boolParam('detailedLegs'),
			withFares: boolParam('withFares'),
			withScheduledSkippedStops: boolParam('withScheduledSkippedStops'),
			numLegAlternatives: parseIntOr(urlParams?.get('numLegAlternatives'), undefined),
			language: getUrlArray('language', [language]),
			transitModes: transitModesUrl.length ? (transitModesUrl as Mode[]) : undefined,
			pedestrianProfile: (urlParams?.get('pedestrianProfile') ?? undefined) as
				| PedestrianProfile
				| undefined,
			useRoutedTransfers: boolParam('useRoutedTransfers'),
			requireBikeTransport: boolParam('requireBikeTransport'),
			requireCarTransport: boolParam('requireCarTransport'),
			preTransitModes: arrParam('preTransitModes') as Mode[] | undefined,
			postTransitModes: arrParam('postTransitModes') as Mode[] | undefined,
			preTransitRentalFormFactors: arrParam('preTransitRentalFormFactors') as
				| RentalFormFactor[]
				| undefined,
			postTransitRentalFormFactors: arrParam('postTransitRentalFormFactors') as
				| RentalFormFactor[]
				| undefined,
			preTransitRentalProviderGroups: arrParam('preTransitRentalProviderGroups'),
			postTransitRentalProviderGroups: arrParam('postTransitRentalProviderGroups'),
			ignorePreTransitRentalReturnConstraints: boolParam('ignorePreTransitRentalReturnConstraints'),
			ignorePostTransitRentalReturnConstraints: boolParam(
				'ignorePostTransitRentalReturnConstraints'
			),
			elevationCosts: (urlParams?.get('elevationCosts') ?? undefined) as ElevationCosts | undefined,
			cyclingSpeed: parseIntOr(urlParams?.get('cyclingSpeed'), undefined),
			pedestrianSpeed: parseIntOr(urlParams?.get('pedestrianSpeed'), undefined),
			maxMatchingDistance: parseIntOr(urlParams?.get('maxMatchingDistance'), undefined),
			maxPreTransitTime: parseIntOr(urlParams?.get('maxPreTransitTime'), undefined),
			maxPostTransitTime: parseIntOr(urlParams?.get('maxPostTransitTime'), undefined)
		});

		const { data: itinerary, error } = await refreshItinerary({ query });
		if (error) {
			console.log(error);
			alert(String((error as Record<string, unknown>).error?.toString() ?? error));
			return;
		}
		updateItinerary(itinerary!, from, to);

		// Populate the search mask (from / to / time) from the reconstructed
		// connection so that editing it (e.g. changing the time) triggers a new
		// plan search via baseQuery, just like a regular search would.
		const legs = itinerary!.legs;
		if (legs.length > 0) {
			const placeToLocation = (p: Place): Location => ({
				label: p.name,
				match: {
					lat: p.lat,
					lon: p.lon,
					level: p.level ?? 0,
					id: p.stopId ?? '',
					areas: [],
					type: p.stopId ? 'STOP' : 'PLACE',
					name: p.name,
					tokens: [],
					score: 0
				}
			});
			from = placeToLocation(legs[0].from);
			to = placeToLocation(legs[legs.length - 1].to);
			time = new Date(arriveBy ? itinerary!.endTime : itinerary!.startTime);

			// Suppress the initial auto-search.
			lastPlanQuery = baseQuery;
		}

		pushStateWithQueryString(
			definedOnly({
				...query,
				fromName: from.label || undefined,
				toName: to.label || undefined
			}),
			{
				selectedItinerary: itinerary,
				selectedStop: replace ? undefined : page.state.selectedStop,
				tripId: replace ? undefined : page.state.tripId,
				activeTab: 'connections'
			},
			replace
		);
	};

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

			const itineraryId = urlParams.get('itineraryId');
			if (itineraryId !== null) {
				onShowItineraryId(itineraryId, true);
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

	let advancedOptionsOpen = $state<boolean>(false);
	let isochronesAdvancedOptionsOpen = $state<boolean>(false);
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
	let viaParam = getUrlArray('via');
	let viaLabels = $state(
		urlParams?.has('viaLabel0')
			? Array.from({ length: viaParam.length }).reduce<Record<string, string>>((acc, _, i) => {
					acc[`viaLabel${i}`] = urlParams?.get(`viaLabel${i}`) ?? '';
					return acc;
				}, {})
			: {}
	);
	let via = $state(
		urlParams?.has('via')
			? viaParam.map((str, i) => parseLocation(str, viaLabels[`viaLabel${i}`]))
			: undefined
	);
	let viaMinimumStay = $state(
		urlParams?.has('via') ? getUrlArray('viaMinimumStay').map((s) => parseIntOr(s, 0)) : undefined
	);
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
		(urlParams?.get('algorithm') ?? 'PONG') as PlanData['query']['algorithm']
	);
	let useRoutedTransfers = $state(
		urlParams?.get('useRoutedTransfers') == 'true' || defaultQuery.useRoutedTransfers
	);
	let pedestrianProfile = $state<PedestrianProfile>(
		(urlParams?.has('pedestrianProfile')
			? urlParams.get('pedestrianProfile')
			: defaultQuery.pedestrianProfile) as PedestrianProfile
	);
	let pedestrianSpeed = $state(
		parseIntOr(urlParams?.get('pedestrianSpeed'), defaultQuery.pedestrianSpeed)
	) as PedestrianSpeed;
	let cyclingSpeed = $state(
		parseIntOr(urlParams?.get('cyclingSpeed'), defaultQuery.cyclingSpeed)
	) as CyclingSpeed;
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
	let transferTimeFactor = $state(
		parseIntOr(urlParams?.get('transferTimeFactor'), defaultQuery.transferTimeFactor)
	);
	let additionalTransferTime = $state(
		parseIntOr(urlParams?.get('additionalTransferTime'), defaultQuery.additionalTransferTime)
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
		errorMessage: undefined,
		errorCode: undefined
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
		from.match && to.match && !advancedOptionsOpen
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
						numLegAlternatives: 3,
						slowDirect,
						fastestDirectFactor: 10,
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
						additionalTransferTime,
						cyclingSpeed,
						pedestrianSpeed,
						transferTimeFactor,
						maxMatchingDistance: pedestrianProfile == 'WHEELCHAIR' ? 8 : 250,
						maxPreTransitTime,
						maxPostTransitTime,
						maxDirectTime,
						ignorePreTransitRentalReturnConstraints,
						ignorePostTransitRentalReturnConstraints,
						ignoreDirectRentalReturnConstraints,
						algorithm,
						via: via ? via.map((v) => v.match?.id) : undefined,
						viaMinimumStay
					} as PlanData['query'])
				} as PlanData)
			: undefined
	);

	let refreshLegAlternativeParams = $derived({
		transitModes: (transitModes.length == possibleTransitModes.length
			? defaultQuery.transitModes
			: transitModes) as Mode[],
		pedestrianProfile,
		useRoutedTransfers,
		requireBikeTransport,
		requireCarTransport,
		preTransitModes: prePostModesToModes(preTransitModes),
		postTransitModes: prePostModesToModes(postTransitModes),
		preTransitRentalFormFactors: getFormFactors(preTransitModes),
		postTransitRentalFormFactors: getFormFactors(postTransitModes),
		preTransitRentalProviderGroups: providerGroupsForQuery(
			preTransitModes,
			preTransitProviderGroups
		),
		postTransitRentalProviderGroups: providerGroupsForQuery(
			postTransitModes,
			postTransitProviderGroups
		),
		ignorePreTransitRentalReturnConstraints,
		ignorePostTransitRentalReturnConstraints,
		elevationCosts,
		cyclingSpeed,
		pedestrianSpeed,
		maxMatchingDistance: pedestrianProfile == 'WHEELCHAIR' ? 8 : 250,
		maxPreTransitTime,
		maxPostTransitTime
	});

	let isochronesQuery = $derived(
		one?.match && !isochronesAdvancedOptionsOpen
			? ({
					query: {
						one: toPlaceString(one),
						maxTravelTime: Math.ceil(maxTravelTime / 60),
						time: time.toISOString(),
						transitModes,
						maxTransfers,
						arriveBy,
						cyclingSpeed,
						pedestrianSpeed,
						transferTimeFactor,
						additionalTransferTime,
						useRoutedTransfers,
						pedestrianProfile,
						requireBikeTransport,
						requireCarTransport,
						preTransitModes: prePostModesToModes(preTransitModes),
						postTransitModes: prePostModesToModes(postTransitModes),
						maxPreTransitTime,
						maxPostTransitTime,
						elevationCosts,
						maxMatchingDistance: pedestrianProfile == 'WHEELCHAIR' ? 8 : 250
					}
				} satisfies Parameters<typeof oneToAll>[0])
			: undefined
	);

	let searchDebounceTimer: number;
	let baseResponse = $state<Promise<PlanResponse>>();
	let routingResponses = $state<Array<Promise<PlanResponse>>>([]);
	let stopNameFromResponse = $state<string>('');
	let refreshingItinerary = $state(false);
	let pageTitle = $derived(
		getPageTitle(
			{
				activeTab,
				from,
				to,
				one,
				selectedStop: page.state.selectedStop,
				stopArriveBy: page.state.stopArriveBy,
				stopName: stopNameFromResponse || page.state.selectedStop?.name,
				selectedItinerary: page.state.selectedItinerary
			},
			t
		)
	);

	const refreshSelectedItinerary = async () => {
		const itineraryId = page.state.selectedItinerary?.id;
		if (!itineraryId || refreshingItinerary) {
			return;
		}

		refreshingItinerary = true;
		try {
			const { data: refreshed, error } = await refreshItinerary({
				query: {
					itineraryId,
					joinInterlinedLegs: false,
					detailedLegs: true,
					detailedTransfers: true,
					withFares: true,
					numLegAlternatives: 3,
					language: [language],
					...refreshLegAlternativeParams
				}
			});

			if (error) {
				console.log(error);
				alert(String((error as Record<string, unknown>).error?.toString() ?? error));
				return;
			}
			if (refreshed && page.state.selectedItinerary?.id === itineraryId) {
				updateItinerary(refreshed, from, to);
				replaceState('', {
					...page.state,
					selectedItinerary: refreshed
				});
			}
		} catch (e) {
			console.log(e);
			alert(String(e));
		} finally {
			refreshingItinerary = false;
		}
	};

	$effect(() => {
		if (baseQuery && baseQuery != lastPlanQuery && activeTab == 'connections') {
			lastPlanQuery = baseQuery;
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
						...(q.toPlace == to.label ? {} : { toName: to.label }),
						...viaLabels
					},
					{ activeTab: 'connections' },
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
				isochronesOptions.errorMessage = undefined;
				isochronesQueryTimeout = setTimeout(async () => {
					try {
						const { data, error, response } = await oneToAll(isochronesQuery);
						if (error) {
							isochronesOptions.status = 'FAILED';
							isochronesOptions.errorCode = response.status;
							isochronesOptions.errorMessage = error.error;
							return;
						}
						const all = data!.all!.map((p: ReachablePlace) => {
							return {
								lat: p.place?.lat,
								lng: p.place?.lon,
								seconds: maxTravelTime - 60 * (p.duration ?? 0),
								name: p.place?.name
							} as IsochronesPos;
						});

						isochronesData = [...all];
						isochronesOptions.status = isochronesData.length == 0 ? 'EMPTY' : 'WORKING';
					} catch (e) {
						isochronesOptions.status = 'FAILED';
						isochronesOptions.errorMessage = String(e);
						isochronesOptions.errorCode = 404;
					}
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
					{ activeTab: 'isochrones' },
					true
				);
			});
		}
	});

	if (browser) {
		addEventListener('paste', (event: ClipboardEvent) => {
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
		map.flyTo({
			...map.cameraForBounds(box, {
				padding: {
					top: 96,
					right: 96,
					bottom: isSmallScreen.current ? window.innerHeight * 0.3 : 96,
					left: isSmallScreen.current ? 96 : 640
				}
			})
		});
	};

	let lastFlownTo: Match | undefined = undefined;
	const flyToLocation = (location: Location) => {
		if (location.match == lastFlownTo) {
			return;
		}
		lastFlownTo = location.match;
		map?.flyTo({ center: location.match, zoom: 18 });
	};

	const flyToSelectedItinerary = () => {
		if (page.state.selectedItinerary && map) {
			flyToItineraries([page.state.selectedItinerary], map);
		}
	};

	$effect(() => {
		if (map) {
			map.addControl(geolocate);
		}
	});

	$effect(() => {
		if (map) {
			if (page.state.selectedItinerary && activeTab == 'connections') {
				flyToSelectedItinerary();
			} else if (activeTab == 'departures' && stop && stop.match) {
				flyToLocation(stop);
			} else if (activeTab == 'isochrones' && one && one.match) {
				flyToLocation(one);
			}
		}
	});

	$effect(() => {
		if (!map || activeTab != 'connections' || !baseQuery) {
			return;
		}
		Promise.all(routingResponses).then((responses) => {
			if (map) {
				let it = responses.flatMap((response) => response.itineraries);
				if (it.length !== 0) {
					flyToItineraries(it, map);
				}
			}
		});
	});
	type CloseFn = () => void;
</script>

<svelte:head>
	<title>{pageTitle}</title>
</svelte:head>

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
		<Tabs.Root
			bind:value={
				() => activeTab,
				(v) => {
					activeTab = v;
					pushState('', { activeTab: v });
				}
			}
			class="max-w-full w-[520px] overflow-y-auto"
		>
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
						bind:advancedOptionsOpen
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
						bind:via
						bind:viaMinimumStay
						bind:viaLabels
						bind:pedestrianSpeed
						bind:cyclingSpeed
						bind:additionalTransferTime
						bind:transferTimeFactor
						{hasDebug}
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
						bind:advancedOptionsOpen={isochronesAdvancedOptionsOpen}
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
						bind:additionalTransferTime
						bind:transferTimeFactor
						bind:cyclingSpeed
						bind:pedestrianSpeed
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
						{hasDebug}
					/>
				</Card>
			</Tabs.Content>
		</Tabs.Root>
	</Control>

	{#if activeTab == 'connections' && routingResponses.length !== 0 && !page.state.selectedItinerary}
		<Control class="min-h-0 md:flex md:flex-col md:mb-2} ">
			<Card
				class="scrollable w-[520px] h-full md:h-[70vh] {isSmallScreen.current
					? 'border-0 shadow-none'
					: ''} overflow-x-hidden bg-background rounded-lg mb-2"
			>
				<ItineraryList
					{baseResponse}
					{routingResponses}
					{baseQuery}
					selectItinerary={(selectedItinerary) => {
						onSelectItinerary(selectedItinerary);
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
								onSelectItinerary(it);
							}}
							{level}
							{theme}
						/>
					{/each}
				{/await}
			{/each}
		{/if}
	{/if}

	{#if activeTab == 'connections' && page.state.selectedItinerary}
		<Control class="min-h-0 md:mb-2 md:flex">
			<Card class="w-[520px] bg-background rounded-lg  flex flex-col mb-2">
				<div class="w-full flex justify-between items-center shadow-md pl-1 mb-1">
					<div class="ml-2 flex items-baseline gap-2">
						<h2 class="text-base font-semibold">{t.journeyDetails}</h2>
						{#if page.state.selectedItinerary.legs.length > 0}
							{@const firstLeg = page.state.selectedItinerary.legs[0]}
							<span class="text-sm text-muted-foreground">
								{formatDate(new Date(firstLeg.startTime), firstLeg.from.tz)}
							</span>
						{/if}
					</div>
					<div class="flex items-center">
						<Button
							variant="ghost"
							size="icon"
							title={t.refreshItinerary}
							aria-label={t.refreshItinerary}
							disabled={refreshingItinerary || !page.state.selectedItinerary.id}
							onclick={refreshSelectedItinerary}
						>
							<RefreshCw class={refreshingItinerary ? 'animate-spin' : ''} />
						</Button>
						<Button
							variant="ghost"
							size="icon"
							onclick={() => {
								history.back();
							}}
						>
							<X />
						</Button>
					</div>
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

	{#if activeTab == 'departures' && page.state.selectedStop}
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
{#if dataLoaded}
	<Map
		bind:map
		bind:bounds
		bind:zoom
		bind:center
		bind:bearing
		class={cn('h-dvh pt-2 overflow-clip', theme)}
		style={showMap ? style : undefined}
		attribution={false}
	>
		{#if hasDebug}
			<Control position="top-right" class="text-right">
				<Debug {bounds} {level} {zoom} />
				<Button
					size="icon"
					variant={showRoutes ? 'default' : 'outline'}
					aria-label="Toggle routes overlay"
					onclick={() => {
						showRoutes = !showRoutes;
					}}
				>
					<Waypoints class="w-5 h-5" />
				</Button>
			</Control>
		{/if}

		<LevelSelect {bounds} {zoom} bind:level />

		{#if browser}
			{#if isSmallScreen.current}
				<Drawer class="relative z-10 h-full mt-3 flex flex-col" bind:showMap>
					{@render resultContent()}
				</Drawer>
			{:else}
				<div class="maplibregl-ctrl-top-left flex flex-col max-h-[97vh]">
					{@render resultContent()}
				</div>
			{/if}
		{/if}

		<div class="maplibregl-ctrl-{isSmallScreen.current ? 'top-left' : 'bottom-right'}">
			<div class="maplibregl-ctrl maplibregl-ctrl-attrib">
				<div class="maplibregl-ctrl-attrib-inner">
					&copy; <a href="http://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>
					{#if withHillshades}
						| <a href="https://mapterhorn.com/attribution" target="_blank">Mapterhorn</a>
					{/if}
					{#if dataAttributionLink}
						| <a href={dataAttributionLink} target="_blank">{t.timetableSources}</a>
					{/if}
				</div>
			</div>
		</div>

		{#if showMap}
			{#if activeTab != 'isochrones'}
				<Control position="top-right" class="w-fit float-right">
					{@const selectedColorMode = colorModeOptions.find((o) => o.value == colorMode)}
					<Select.Root type="single" bind:value={colorMode} items={colorModeOptions}>
						<Select.Trigger class="bg-background w-40 gap-2">
							{#if selectedColorMode}
								{@const Icon = selectedColorMode.icon}
								<Icon class="h-[1.2rem] w-[1.2rem]" />
								<span class="grow text-left">{selectedColorMode.label}</span>
							{/if}
						</Select.Trigger>
						<Select.Content align="end">
							{#each colorModeOptions as option (option.value)}
								{@const Icon = option.icon}
								<Select.Item value={option.value} label={option.label} class="gap-2">
									<Icon class="h-[1.2rem] w-[1.2rem]" />
									{option.label}
								</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</Control>
				<Control position="top-right" class="w-fit float-right pb-4">
					<Button
						class={bearing === 0 ? 'hidden' : null}
						size="icon"
						title={t.resetToNorth}
						onclick={() => map!.resetNorth()}
					>
						<Compass class="w-5 h-5" />
					</Button>
					<Button size="icon" title={t.showMyLocation} onclick={() => getLocation()}>
						<LocateFixed class="w-5 h-5" />
					</Button>
					<Button
						size="icon"
						title={t.toggleHillshades}
						variant={withHillshades ? 'default' : 'outline'}
						onclick={() => (withHillshades = !withHillshades)}
					>
						<MountainSnow class="w-5 h-5" />
					</Button>
				</Control>
				{#if showRoutes}
					<Routes
						{map}
						{bounds}
						{zoom}
						shapesDebugEnabled={serverConfig?.shapesDebugEnabled === true}
					/>
				{/if}
				<Rentals
					{map}
					{bounds}
					{zoom}
					{theme}
					isSmallScreen={isSmallScreen.current}
					debug={hasDebug}
				/>
			{/if}

			{#if colorMode === 'stops'}
				<StopsView {map} {bounds} {zoom} {theme} />
			{/if}
			<RailViz
				{map}
				{bounds}
				{zoom}
				colorMode={colorMode === 'rt' || colorMode === 'route' || colorMode === 'mode'
					? colorMode
					: 'none'}
			/>
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

			{#if stop && activeTab == 'departures'}
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
				<Marker
					color="yellow"
					draggable={true}
					{level}
					bind:location={one}
					bind:marker={oneMarker}
				/>
			{/if}
		{/if}
	</Map>
{/if}
