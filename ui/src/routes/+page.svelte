<script lang="ts">
	import X from 'lucide-svelte/icons/x';
	import { getStyle } from '$lib/map/style';
	import Map from '$lib/map/Map.svelte';
	import Control from '$lib/map/Control.svelte';
	import SearchMask from './SearchMask.svelte';
	import { posToLocation, type Location } from '$lib/Location';
	import { Card } from '$lib/components/ui/card';
	import {
		initial,
		type Itinerary,
		plan,
		type PlanResponse,
		railviz,
		trip,
		type TripSegment
	} from '$lib/openapi';
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
	import { formatTime, toDateTime } from '$lib/toDateTime';
	import { onMount } from 'svelte';

	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import { createTripIcon } from '$lib/map/createTripIcon';
	import { getColor } from '$lib/modeStyle';
	import getDistance from '@turf/rhumb-distance';
	import getBearing from '@turf/rhumb-bearing';
	import polyline from 'polyline';

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

	function hexToRgb(hex: string): [number, number, number, number] {
		var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		if (!result) {
			throw `${hex} is not a hex color #RRGGBB`;
		}
		return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16), 255];
	}

	type KeyFrame = { point: [number, number]; heading: number; time: number };

	const getKeyFrames = (t: TripSegment): Array<KeyFrame> => {
		let keyFrames: Array<KeyFrame> = [];
		const coordinates = polyline.decode(t.polyline).map(([x, y]): [number, number] => [y, x]);
		const totalDistance = t.distance;
		const totalDuration = t.arrival - t.departure;
		let currDistance = 0;
		for (let i = 0; i < coordinates.length - 1; i++) {
			let from = coordinates[i];
			let to = coordinates[i + 1];

			const distance = getDistance(from, to, { units: 'kilometers' }) * 1000;
			const heading = getBearing(from, to);

			const r = currDistance / totalDistance;
			keyFrames.push({ point: from, heading, time: t.departure + r * totalDuration });

			currDistance += distance;
		}
		keyFrames.push({ point: coordinates[coordinates.length - 1], time: t.arrival, heading: 0 });
		return keyFrames;
	};

	const getFrame = (keyframes: Array<KeyFrame>, timestamp: number) => {
		const i = keyframes.findIndex((s) => s.time >= timestamp);

		if (i === -1 || i === 0) {
			return;
		}

		const startState = keyframes[i - 1];
		const endState = keyframes[i];
		const r = (timestamp - startState.time) / (endState.time - startState.time);

		return {
			point: [
				startState.point[0] * (1 - r) + endState.point[0] * r,
				startState.point[1] * (1 - r) + endState.point[1] * r
			],
			heading: startState.heading
		};
	};

	const getRailvizLayer = (trips: Array<TripSegment & { keyFrames: Array<KeyFrame> }>) => {
		const now = new Date().getTime();

		const tripsWithFrame = trips
			.map((t) => {
				return {
					...t,
					...getFrame(t.keyFrames, now)
				};
			})
			.filter((t) => t.point);

		return new IconLayer<TripSegment & { keyFrames: Array<KeyFrame> } & KeyFrame>({
			id: 'trips',
			data: tripsWithFrame,
			beforeId: 'road-name-text',
			getColor: (d) => hexToRgb(getColor(d)[0]),
			getAngle: (d) => -d.heading + 90,
			getPosition: (d) => d.point,
			getSize: (d) => 48,
			getIcon: (_) => 'marker',
			pickable: true,
			// @ts-expect-error: canvas element seems to work fine
			iconAtlas: createTripIcon(128),
			iconMapping: {
				marker: {
					x: 0,
					y: 0,
					width: 128,
					height: 128,
					anchorY: 64,
					anchorX: 64,
					mask: true
				}
			}
		});
	};

	const railvizRequest = () => {
		const b = maplibregl.LngLatBounds.convert(bounds!);
		const min = lngLatToStr(b.getNorthWest());
		const max = lngLatToStr(b.getSouthEast());
		const startTime = new Date().getTime() / 1000;
		const endTime = startTime + 2 * 60;
		return railviz({
			query: {
				min,
				max,
				startTime,
				endTime,
				zoom
			}
		});
	};

	let animation: number | null = null;
	const updateRailvizLayer = () => {
		railvizRequest().then((d) => {
			if (animation) {
				cancelAnimationFrame(animation);
			}

			const tripSegmentsWithKeyFrames = d.data!.map((tripSegment: TripSegment) => {
				return { ...tripSegment, keyFrames: getKeyFrames(tripSegment) };
			});

			const onAnimationFrame = () => {
				overlay!.setProps({
					layers: [getRailvizLayer(tripSegmentsWithKeyFrames)]
				});
				animation = requestAnimationFrame(onAnimationFrame);
			};

			onAnimationFrame();
		});
	};

	let timer: number | undefined;
	let overlay = $state.raw<MapboxOverlay>();
	$effect(() => {
		if (map && !overlay) {
			overlay = new MapboxOverlay({
				interleaved: true,
				layers: [],
				getTooltip: ({ object }) => {
					if (!object) {
						return null;
					}
					return {
						className: 'bg-red-500',
						html: `${object.trips[0].routeShortName}<br>
										${formatTime(new Date(object.departure))} ${object.from.name}<br>
										${formatTime(new Date(object.arrival))} ${object.to.name}`
					};
				},
				onClick: ({ object }) => {
					onClickTrip(object.trips[0].tripId, object.trips[0].serviceDate);
				}
			});
			map.addControl(overlay);
			updateRailvizLayer();
			timer = setTimeout(updateRailvizLayer, 1000);
		}
	});

	$effect(() => {
		if (overlay && bounds && zoom) {
			updateRailvizLayer();
			clearTimeout(timer);
			timer = setTimeout(updateRailvizLayer, 1000);
		}
	});

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
