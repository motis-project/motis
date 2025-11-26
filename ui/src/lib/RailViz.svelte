<script lang="ts">
	import { trips, type Mode, type TripSegment } from '@motis-project/motis-client';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import { createTripIcon } from '$lib/map/createTripIcon';
	import { getColor, getModeStyle } from '$lib/modeStyle';
	import getDistance from '@turf/rhumb-distance';
	import getBearing from '@turf/rhumb-bearing';
	import polyline from '@mapbox/polyline';
	import { formatTime } from '$lib/toDateTime';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import maplibregl from 'maplibre-gl';
	import { onDestroy, untrack } from 'svelte';
	import Control from '$lib/map/Control.svelte';
	import { onClickTrip } from '$lib/utils';

	let {
		map,
		bounds,
		zoom,
		colorMode
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		colorMode: 'rt' | 'route' | 'mode' | 'none';
	} = $props();

	let railvizError = $state();

	type RGBA = [number, number, number, number];

	function hexToRgb(hex: string): RGBA {
		var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
		if (!result) {
			throw `${hex} is not a hex color #RRGGBB`;
		}
		return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16), 255];
	}

	function rgbToHex(rgba: RGBA): string {
		return '#' + ((1 << 24) | (rgba[0] << 16) | (rgba[1] << 8) | rgba[2]).toString(16).slice(1);
	}

	const getDelayColor = (delay: number, realTime: boolean): RGBA => {
		delay = delay / 60000;
		if (!realTime) {
			return [100, 100, 100, 255];
		}
		if (delay <= -5) {
			return [255, 0, 255, 255];
		} else if (delay <= -1) {
			return [138, 82, 254, 255];
		} else if (delay <= 3) {
			return [69, 194, 74, 255];
		} else if (delay <= 5) {
			return [255, 237, 0, 255];
		} else if (delay <= 10) {
			return [255, 102, 0, 255];
		} else if (delay <= 15) {
			return [255, 48, 71, 255];
		}
		return [163, 0, 10, 255];
	};

	const getSegmentDelayColor = (d: number, a: number, realTime: boolean): RGBA => {
		if (d / 60000 <= -1) {
			return getDelayColor(d, realTime);
		} else {
			return getDelayColor(a, realTime);
		}
	};

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
		departureDelay: number;
	};

	const getKeyFrames = (t: TripSegment): KeyFrameExt => {
		let keyFrames: Array<KeyFrame> = [];
		const departure = new Date(t.departure).getTime();
		const arrival = new Date(t.arrival).getTime();
		const scheduledArrival = new Date(t.scheduledArrival).getTime();
		const scheduledDeparture = new Date(t.scheduledDeparture).getTime();
		const arrivalDelay = arrival - scheduledArrival;
		const departureDelay = departure - scheduledDeparture;
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
			console.debug(totalDistance, currDistance);
		}
		keyFrames.push({ point: coordinates[coordinates.length - 1], time: arrival, heading: 0 });
		return { keyFrames, arrival, departure, arrivalDelay, departureDelay };
	};

	const getFrame = (keyframes: Array<KeyFrame>, timestamp: number) => {
		const i = keyframes.findIndex((s) => s.time >= timestamp);

		if (i === -1 || i === 0) {
			console.debug(
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
		trips: Array<{ realTime: boolean; arrivalDelay: number; departureDelay: number } & KeyFrameExt>
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

		return new IconLayer<
			{
				realTime: boolean;
				arrivalDelay: number;
				departureDelay: number;
				routeColor?: string;
				routeTextColor?: string;
				mode: Mode;
			} & KeyFrame
		>({
			id: 'trips',
			data: tripsWithFrame,
			beforeId: 'road-name-text',
			getColor: (d) => {
				switch (colorMode) {
					case 'rt':
						return getSegmentDelayColor(d.departureDelay, d.arrivalDelay, d.realTime);
					case 'mode':
						return hexToRgb(getModeStyle(d)[1]);
					case 'route':
						return hexToRgb(getColor(d)[0]);
					case 'none':
						return hexToRgb(getColor(d)[0]);
				}
			},
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
			if (colorMode == 'none') {
				if (animation) {
					cancelAnimationFrame(animation);
				}
				overlay!.setProps({
					layers: []
				});
				clearTimeout(timer);
				return;
			}
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
			console.debug('updateRailviz: timer');
			updateRailviz();
		}, 60000);
	};

	const tooltipPopup = new maplibregl.Popup({
		closeButton: false,
		closeOnClick: false,
		maxWidth: 'none'
	});

	$effect(() => {
		if (map && !overlay) {
			overlay = new MapboxOverlay({
				interleaved: true,
				layers: [],
				onClick: ({ object }) => {
					if (!object) {
						return;
					}
					onClickTrip(object.trips[0].tripId);
				},
				getCursor: () => map.getCanvas().style.cursor,
				onHover: ({ object, coordinate }) => {
					if (object && coordinate) {
						const popup = tooltipPopup.setLngLat(coordinate as [number, number]);
						if (object.realTime) {
							popup.setHTML(
								`<strong>${object.trips[0].displayName}</strong><br>

							<span style="color: ${rgbToHex(getDelayColor(object.departureDelay, true))}">${formatTime(new Date(object.departure), object.from.tz)}</span>
							<span class="line-through">${formatTime(new Date(object.scheduledDeparture), object.from.tz)}</span>  ${object.from.name}<br>

							<span style="color: ${rgbToHex(getDelayColor(object.arrivalDelay, true))}">${formatTime(new Date(object.arrival), object.to.tz)}</span>
							<span class="line-through">${formatTime(new Date(object.scheduledArrival), object.to.tz)}</span> ${object.to.name}`
							);
						} else {
							popup.setHTML(
								`<strong>${object.trips[0].displayName}</strong><br>
							${formatTime(new Date(object.departure), object.from.tz)} ${object.from.name}<br>
							${formatTime(new Date(object.arrival), object.to.tz)} ${object.to.name}`
							);
						}
						popup.addTo(map);
					} else {
						tooltipPopup.remove();
					}
				}
			});
			map.addControl(overlay);

			console.debug('updateRailviz: init');
			untrack(() => updateRailviz());
		}
	});

	$effect(() => {
		if (overlay && bounds && zoom && colorMode) {
			untrack(() => {
				console.debug(`updateRailviz: effect ${overlay} ${bounds} ${zoom} ${colorMode}`);
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

{#if railvizError}
	<Control position="bottom-left">
		{railvizError}
	</Control>
{/if}
