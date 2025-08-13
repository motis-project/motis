<script lang="ts">
	import { trips, type Mode, type TripSegment } from '$lib/api/openapi';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import { createTripIcon } from '$lib/map/createTripIcon';
	import { getColor } from '$lib/modeStyle';
	import getDistance from '@turf/rhumb-distance';
	import getBearing from '@turf/rhumb-bearing';
	import polyline from '@mapbox/polyline';
	import { formatTime } from '$lib/toDateTime';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import maplibregl from 'maplibre-gl';
	import { onDestroy, untrack } from 'svelte';
	import Control from '$lib/map/Control.svelte';
	import { Button } from '$lib/components/ui/button';
	import Palette from 'lucide-svelte/icons/palette';
	import Rss from 'lucide-svelte/icons/rss';
	import LocateFixed from 'lucide-svelte/icons/locate-fixed';
	import { browser } from '$app/environment';
	import { onClickTrip } from '$lib/utils';

	let {
		map,
		bounds,
		zoom
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
	} = $props();

	let colorMode = $state<'rt' | 'route'>('route');
	let railvizError = $state();

	type RGBA = [number, number, number, number];

	function hexToRgb(hex: string): RGBA {
		var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		if (!result) {
			throw `${hex} is not a hex color #RRGGBB`;
		}
		return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16), 255];
	}

	type KeyFrame = {
		point: [number, number];
		heading: number;
		time: number;
	};

	type KeyFrameExt = {
		keyFrames: Array<KeyFrame>;
		arrival: number;
		departure: number;
		arrivalDelay: number;
	};

	const getKeyFrames = (t: TripSegment): KeyFrameExt => {
		let keyFrames: Array<KeyFrame> = [];
		const departure = new Date(t.departure).getTime();
		const arrival = new Date(t.arrival).getTime();
		const scheduledArrival = new Date(t.scheduledArrival).getTime();
		const arrivalDelay = arrival - scheduledArrival;
		const coordinates = polyline.decode(t.polyline).map(([x, y]): [number, number] => [y, x]);
		const totalDuration = arrival - departure;
		let currDistance = 0;

		let totalDistance = 0;
		for (let i = 0; i < coordinates.length - 1; i++) {
			let from = coordinates[i];
			let to = coordinates[i + 1];
			totalDistance += getDistance(from, to, { units: 'meters' });
		}

		for (let i = 0; i < coordinates.length - 1; i++) {
			let from = coordinates[i];
			let to = coordinates[i + 1];

			const distance = getDistance(from, to, { units: 'meters' });
			const heading = getBearing(from, to);

			const r = currDistance / totalDistance;
			keyFrames.push({ point: from, heading, time: departure + r * totalDuration });

			currDistance += distance;
		}
		if (Math.abs(totalDistance - currDistance) > 1) {
			console.log(totalDistance, currDistance);
		}
		keyFrames.push({ point: coordinates[coordinates.length - 1], time: arrival, heading: 0 });
		return { keyFrames, arrival, departure, arrivalDelay };
	};

	const getFrame = (keyframes: Array<KeyFrame>, timestamp: number) => {
		const i = keyframes.findIndex((s) => s.time >= timestamp);

		if (i === -1 || i === 0) {
			console.log(
				'not found, timestamp=',
				new Date(timestamp),
				' #keyframes=',
				keyframes.length,
				' first=',
				new Date(keyframes[0].time),
				', last=',
				new Date(keyframes[keyframes.length - 1].time)
			);
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

	const getRailvizLayer = (
		trips: Array<{ realTime: boolean; arrivalDelay: number } & KeyFrameExt>
	) => {
		const now = new Date().getTime();

		const tripsWithFrame = trips
			.filter((t) => now >= t.departure && now <= t.arrival)
			.map((t) => {
				return {
					...t,
					...getFrame(t.keyFrames, now)
				};
			})
			.filter((t) => t.point);

		const getDelayColor = (d: number, realTime: boolean): RGBA => {
			const delay = d / 60000;
			if (!realTime) {
				return [100, 100, 100, 255];
			}
			if (delay <= 3) {
				return [69, 209, 74, 255];
			} else if (delay <= 5) {
				return [255, 237, 0, 255];
			} else if (delay <= 10) {
				return [255, 102, 0, 255];
			} else if (delay <= 15) {
				return [255, 48, 71, 255];
			}
			return [163, 0, 10, 255];
		};

		return new IconLayer<
			{
				realTime: boolean;
				arrivalDelay: number;
				routeColor?: string;
				routeTextColor?: string;
				mode: Mode;
			} & KeyFrame
		>({
			id: 'trips',
			data: tripsWithFrame,
			beforeId: 'road-name-text',
			getColor: (d) =>
				colorMode == 'rt' ? getDelayColor(d.arrivalDelay, d.realTime) : hexToRgb(getColor(d)[0]),
			getAngle: (d) => -d.heading + 90,
			getPosition: (d) => d.point,
			getSize: (_) => 48,
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
		const startTime = new Date();
		const endTime = new Date(startTime.getTime() + 180000);
		return trips({
			query: {
				min,
				max,
				startTime: startTime.toISOString(),
				endTime: endTime.toISOString(),
				zoom
			}
		});
	};

	let animation: number | null = null;
	const updateRailvizLayer = async () => {
		try {
			const { data, error, response } = await railvizRequest();
			if (animation) {
				cancelAnimationFrame(animation);
			}

			if (error) {
				railvizError = `map trips error status ${response.status}`;
				return;
			}

			railvizError = undefined;

			const tripSegmentsWithKeyFrames = data!.map((tripSegment: TripSegment) => {
				return { ...tripSegment, ...getKeyFrames(tripSegment) };
			});

			const onAnimationFrame = () => {
				overlay!.setProps({
					layers: [getRailvizLayer(tripSegmentsWithKeyFrames)]
				});
				animation = requestAnimationFrame(onAnimationFrame);
			};

			onAnimationFrame();
		} catch (e) {
			railvizError = e;
		}
	};

	let timer: number | undefined;
	let overlay = $state.raw<MapboxOverlay>();
	const updateRailviz = async () => {
		await updateRailvizLayer();
		clearTimeout(timer); // Ensure previous timer is cleared
		timer = setTimeout(() => {
			console.log('updateRailviz: timer');
			updateRailviz();
		}, 60000);
	};

	const geolocate = new maplibregl.GeolocateControl({
		positionOptions: {
			enableHighAccuracy: true
		},
		showAccuracyCircle: false
	});

	const getLocation = () => {
		geolocate.trigger();
	};

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
						html: `${object.trips[0].displayName}<br>
                  ${formatTime(new Date(object.departure))} ${object.from.name}<br>
                  ${formatTime(new Date(object.arrival))} ${object.to.name}`
					};
				},
				onClick: ({ object }) => {
					if (!object) {
						return;
					}
					onClickTrip(object.trips[0].tripId);
				}
			});
			map.addControl(geolocate);
			map.addControl(overlay);

			console.log('updateRailviz: init');
			untrack(() => updateRailviz());
		}
	});

	$effect(() => {
		if (overlay && bounds && zoom && colorMode) {
			untrack(() => {
				console.log(`updateRailviz: effect ${overlay} ${bounds} ${zoom} ${colorMode}`);
				updateRailviz();
			});
		}
	});

	onDestroy(() => {
		if (animation) {
			cancelAnimationFrame(animation);
		}
		clearTimeout(timer);
		if (map && overlay) {
			map.removeControl(overlay);
		}
	});
</script>

<Control position={browser && window.innerWidth < 768 ? 'bottom-left' : 'top-right'} class="pb-4">
	<Button
		size="icon"
		variant={colorMode ? 'default' : 'outline'}
		onclick={() => {
			colorMode = colorMode == 'rt' ? 'route' : 'rt';
		}}
	>
		{#if colorMode == 'rt'}
			<Rss class="h-[1.2rem] w-[1.2rem]" />
		{:else}
			<Palette class="h-[1.2rem] w-[1.2rem]" />
		{/if}
	</Button>
	<Button size="icon" onclick={() => getLocation()}>
		<LocateFixed class="w-5 h-5" />
	</Button>
</Control>

{#if railvizError}
	<Control position="bottom-left">
		{railvizError}
	</Control>
{/if}
