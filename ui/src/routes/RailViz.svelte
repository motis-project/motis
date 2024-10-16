<script lang="ts">
	import { trips, type TripSegment } from '$lib/openapi';
	import { MapboxOverlay } from '@deck.gl/mapbox';
	import { IconLayer } from '@deck.gl/layers';
	import { createTripIcon } from '$lib/map/createTripIcon';
	import { getColor } from '$lib/modeStyle';
	import getDistance from '@turf/rhumb-distance';
	import getBearing from '@turf/rhumb-bearing';
	import polyline from 'polyline';
	import { formatTime } from '$lib/toDateTime';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import maplibregl from 'maplibre-gl';
	import { onDestroy } from 'svelte';

	let {
		map,
		bounds,
		zoom,
		onClickTrip
	}: {
		map: maplibregl.Map | undefined;
		bounds: maplibregl.LngLatBoundsLike | undefined;
		zoom: number;
		onClickTrip: (tripId: string, date: string) => void;
	} = $props();

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

	const getKeyFrames = (t: TripSegment): Array<KeyFrame> => {
		let keyFrames: Array<KeyFrame> = [];
		const coordinates = polyline.decode(t.polyline).map(([x, y]): [number, number] => [y, x]);
		const totalDuration = t.arrival - t.departure;
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
			.filter((t) => now >= t.departure && now < t.arrival)
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

		const colorMode = 'realtime';

		return new IconLayer<TripSegment & { keyFrames: Array<KeyFrame> } & KeyFrame>({
			id: 'trips',
			data: tripsWithFrame,
			beforeId: 'road-name-text',
			getColor: (d) =>
				colorMode == 'realtime'
					? getDelayColor(d.arrivalDelay, d.realTime)
					: hexToRgb(getColor(d)[0]),
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
		const startTime = new Date().getTime() / 1000;
		const endTime = startTime + 2 * 60;
		return trips({
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
	const updateRailviz = () => {
		clearTimeout(timer);
		updateRailvizLayer();
		timer = setTimeout(updateRailviz, 60000);
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
						html: `${object.trips[0].routeShortName}<br>
                  ${formatTime(new Date(object.departure))} ${object.from.name}<br>
                  ${formatTime(new Date(object.arrival))} ${object.to.name}`
					};
				},
				onClick: ({ object }) => {
					if (!object) {
						return;
					}
					onClickTrip(object.trips[0].tripId, object.trips[0].serviceDate);
				}
			});
			map.addControl(overlay);

			updateRailviz();
		}
	});

	$effect(() => {
		if (overlay && bounds && zoom) {
			updateRailviz();
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
