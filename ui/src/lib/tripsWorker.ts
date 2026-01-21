/// <reference lib="webworker" />trips
import polyline from '@mapbox/polyline';
import { client, trips, type TripSegment } from '@motis-project/motis-client';
import type { Trip, TransferData, MetaData, Query } from './types';
import type { QuerySerializerOptions } from '@hey-api/client-fetch';
import { getDelayColor, hexToRgb } from './Color';
import { getModeStyle, getColor } from './modeStyle';

const concatFloat32Arrays = (a: Float32Array, b: Float32Array): Float32Array => {
	const res = new Float32Array(a.length + b.length);
	res.set(a);
	res.set(b, a.length);
	return res;
};
const concatFloat64Arrays = (a: Float64Array, b: Float64Array): Float64Array => {
	const res = new Float64Array(a.length + b.length);
	res.set(a);
	res.set(b, a.length);
	return res;
};

//MATH
const ease = (t: number) => t * t * (3 - 2 * t);
const R = 6371;
const TO_RAD = Math.PI / 180;
const TO_DEG = 180 / Math.PI;

const getSpacialData = (
	path: Float64Array,
	i: number,
	segmentDistances: Float64Array,
	headings: Float32Array
) => {
	const lon2 = path[i * 2] * TO_RAD;
	const lat2 = path[i * 2 + 1] * TO_RAD;
	const lon1 = path[(i - 1) * 2] * TO_RAD;
	const lat1 = path[(i - 1) * 2 + 1] * TO_RAD;

	const dLon = lon2 - lon1;
	const dLat = lat2 - lat1;

	const cosLat1 = Math.cos(lat1);
	const cosLat2 = Math.cos(lat2);
	const sinLat1 = Math.sin(lat1);
	const sinLat2 = Math.sin(lat2);
	const cosDLon = Math.cos(dLon);

	// Haversine Distance
	const a = Math.sin(dLat / 2) ** 2 + cosLat1 * cosLat2 * Math.sin(dLon / 2) ** 2;
	const dist = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) * R;
	segmentDistances[i - 1] = dist;

	// Bearing
	const y = Math.sin(dLon) * cosLat2;
	const x = cosLat1 * sinLat2 - sinLat1 * cosLat2 * cosDLon;
	headings[i - 1] = (-(Math.atan2(y, x) * TO_DEG + 360) % 360) + 90;
};

//PROCESSING
let status: number;
const tripsMap = new Map<string, Trip>();
const metaDataMap = new Map<string, MetaData>();
let metadata: MetaData[] = [];
const fetchData = async (query: Query) => {
	const { data, response } = await trips({ query });
	status = response.status;
	if (!data) return;
	tripsMap.clear();
	metaDataMap.clear();
	data.forEach((d) => {
		const id = d.trips[0].tripId;
		const processed = processSegment(d);
		const existing = tripsMap.get(id);
		if (existing) {
			if (existing.timestamps[0] >= processed.timestamps[processed.timestamps.length - 1]) {
				existing.timestamps = concatFloat64Arrays(processed.timestamps, existing.timestamps);
				existing.path = concatFloat64Arrays(processed.path, existing.path);
				existing.headings = concatFloat32Arrays(processed.headings, existing.headings);
			} else {
				existing.timestamps = concatFloat64Arrays(existing.timestamps, processed.timestamps);
				existing.headings = concatFloat32Arrays(existing.headings, processed.headings);
				existing.path = concatFloat64Arrays(existing.path, processed.path);
			}
		} else {
			tripsMap.set(id, processed);
		}
		metaDataMap.set(id, {
			id: id,
			displayName: d.trips[0].displayName,
			tz: d.from.tz,
			from: d.from.name,
			to: d.to.name,
			realtime: d.realTime,
			arrival: d.arrival,
			departure: d.departure,
			scheduledArrival: d.scheduledArrival,
			scheduledDeparture: d.scheduledDeparture,
			departureDelay: processed.departureDelay,
			arrivalDelay: processed.arrivalDelay
		});
	});
	metadata = Array.from(metaDataMap.values());
};
const processSegment = (s: TripSegment): Trip => {
	const departure = new Date(s.departure).getTime();
	const arrival = new Date(s.arrival).getTime();
	const totalDuration = arrival - departure;

	const decoded = polyline.decode(s.polyline);
	const count = decoded.length;

	const path = new Float64Array(count * 2);
	const timestamps = new Float64Array(count);
	const headings = new Float32Array(count);
	const segmentDistances = new Float64Array(count);

	let totalDistance = 0;

	for (let i = 0; i < count; i++) {
		const lon = decoded[i][1];
		const lat = decoded[i][0];
		path[i * 2] = lon;
		path[i * 2 + 1] = lat;

		if (i > 0) {
			getSpacialData(path, i, segmentDistances, headings);
			totalDistance += segmentDistances[i - 1];
		}
	}

	if (count > 0) headings[count - 1] = headings[count - 2] || 0;

	const invTotalDist = totalDistance === 0 ? 0 : 1 / totalDistance;
	let cumulativeDist = 0;
	for (let i = 0; i < count; i++) {
		timestamps[i] = departure + cumulativeDist * invTotalDist * totalDuration;
		cumulativeDist += segmentDistances[i];
	}

	return {
		realtime: s.realTime,
		mode: s.mode,
		routeColor: s.routeColor,
		path,
		timestamps,
		headings,
		currentIndx: 0,
		departureDelay: departure - new Date(s.scheduledDeparture).getTime(),
		arrivalDelay: arrival - new Date(s.scheduledArrival).getTime()
	};
};

//STATE UPDATE
function updateState(data: TransferData, colorMode: string) {
	let posIndex = 0;
	let colorIndex = 0;
	let angleIndex = 0;
	const time = Date.now();
	let color;
	tripsMap.forEach((t) => {
		const stamps = t.timestamps;
		const path = t.path;
		const headings = t.headings;
		const len = stamps.length;

		switch (colorMode) {
			case 'rt':
				color = getDelayColor(t.departureDelay, t.realtime);
				break;
			case 'mode':
				color = hexToRgb(getModeStyle(t)[1]);
				break;
			case 'route':
				color = hexToRgb(getColor(t)[0]);
				break;
			case 'none':
				color = hexToRgb(getColor(t)[0]);
				break;
		}
		data.colors[colorIndex] = color![0];
		data.colors[colorIndex + 1] = color![1];
		data.colors[colorIndex + 2] = color![2];

		let curr = t.currentIndx;
		while (curr < len - 1 && stamps[curr] < time) {
			curr++;
		}
		t.currentIndx = curr;

		const last = stamps[len - 1];

		if (curr === 0) {
			data.positions[posIndex] = path[0];
			data.positions[posIndex + 1] = path[1];
			data.angles[angleIndex] = headings[0];
		} else if (curr === len - 1 && time >= last) {
			const idx = 2 * (len - 1);
			data.positions[posIndex] = path[idx];
			data.positions[posIndex + 1] = path[idx + 1];
			data.angles[angleIndex] = headings[len - 1];
		} else if (last > time) {
			const t0 = stamps[curr - 1];
			const t1 = stamps[curr];

			const r = ease((time - t0) / (t1 - t0));

			const prevIdx = 2 * (curr - 1);
			const nextIdx = 2 * curr;

			const x0 = path[prevIdx];
			const y0 = path[prevIdx + 1];

			data.positions[posIndex] = x0 + (path[nextIdx] - x0) * r;
			data.positions[posIndex + 1] = y0 + (path[nextIdx + 1] - y0) * r;
			data.angles[angleIndex] = headings[curr - 1];
		}

		colorIndex += 3;
		angleIndex++;
		posIndex += 2;
	});

	data.length = angleIndex;
}

//MESSAGING
self.onmessage = async (e) => {
	switch (e.data.type) {
		case 'init': {
			const querySerializer = { array: { explode: false } } as QuerySerializerOptions;
			client.setConfig({ baseUrl: e.data.origin, querySerializer });
			break;
		}
		case 'fetch':
			await fetchData(e.data.query);
			postMessage({ type: 'fetch-complete', status });
			break;
		case 'update': {
			const positions = new Float64Array(e.data.positions.buffer);
			const angles = new Float32Array(e.data.angles.buffer);
			const colors = new Uint8Array(e.data.colors.buffer);
			const hovIndex = e.data.index;
			const data = {
				colors,
				positions,
				angles,
				length: 0
			};
			updateState(data, e.data.colorMode);
			postMessage(
				{
					angles: data.angles,
					positions: data.positions,
					colors: data.colors,
					length: data.length,
					metadata: hovIndex !== -1 ? metadata[hovIndex] : null
				},
				[data.angles.buffer, data.positions.buffer, data.colors.buffer]
			);
			break;
		}
	}
};

export {};
