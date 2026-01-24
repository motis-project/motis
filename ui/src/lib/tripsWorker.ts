/// <reference lib="webworker" />trips
import polyline from '@mapbox/polyline';
import { client, trips, type Mode, type TripSegment } from '@motis-project/motis-client';
import type { Trip, TransferData, MetaData, Query } from './types';
import type { QuerySerializerOptions } from '@hey-api/client-fetch';
import { getDelayColor, hexToRgb } from './Color';
import { getModeStyle, getColor } from './modeStyle';

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
	const i2 = i * 2;
	const i1 = i2 - 2;
	const lon2 = path[i2] * TO_RAD;
	const lat2 = path[i2 + 1] * TO_RAD;
	const lon1 = path[i1] * TO_RAD;
	const lat1 = path[i1 + 1] * TO_RAD;

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

	const tripBuilders = new Map<
		string,
		{
			paths: Float64Array[];
			timestamps: Float64Array[];
			headings: Float32Array[];
			realtime: boolean;
			mode: Mode;
			routeColor?: string;
			departureDelay: number;
			arrivalDelay: number;
		}
	>();

	for (const d of data) {
		const id = d.trips[0].tripId;
		const processed = processSegment(d);

		if (!tripBuilders.has(id)) {
			tripBuilders.set(id, {
				paths: [processed.path],
				timestamps: [processed.timestamps],
				headings: [processed.headings],
				realtime: processed.realtime,
				mode: processed.mode,
				routeColor: processed.routeColor,
				departureDelay: processed.departureDelay,
				arrivalDelay: processed.arrivalDelay
			});
		} else {
			const b = tripBuilders.get(id)!;
			b.paths.push(processed.path);
			b.timestamps.push(processed.timestamps);
			b.headings.push(processed.headings);
		}

		metaDataMap.set(id, {
			id,
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
	}

	tripBuilders.forEach((b, id) => {
		let pathLen = 0;
		let tsLen = 0;
		let hdLen = 0;

		for (let i = 0; i < b.paths.length; i++) {
			pathLen += b.paths[i].length;
			tsLen += b.timestamps[i].length;
			hdLen += b.headings[i].length;
		}

		const path = new Float64Array(pathLen);
		const timestamps = new Float64Array(tsLen);
		const headings = new Float32Array(hdLen);

		let pOff = 0,
			tOff = 0,
			hOff = 0;
		for (let i = 0; i < b.paths.length; i++) {
			path.set(b.paths[i], pOff);
			timestamps.set(b.timestamps[i], tOff);
			headings.set(b.headings[i], hOff);

			pOff += b.paths[i].length;
			tOff += b.timestamps[i].length;
			hOff += b.headings[i].length;
		}

		tripsMap.set(id, {
			realtime: b.realtime,
			mode: b.mode,
			routeColor: b.routeColor,
			path,
			timestamps,
			headings,
			currentIndx: 0,
			departureDelay: b.departureDelay,
			arrivalDelay: b.arrivalDelay
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
		const p = decoded[i];
		const lon = p[1];
		const lat = p[0];
		path[i * 2] = lon;
		path[i * 2 + 1] = lat;

		if (i > 0) {
			getSpacialData(path, i, segmentDistances, headings);
			totalDistance += segmentDistances[i - 1];
		}
	}

	if (count > 1) headings[count - 1] = headings[count - 2];

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

			const x1 = path[nextIdx];
			const y1 = path[nextIdx + 1];

			data.positions[posIndex] = x0 + (x1 - x0) * r;
			data.positions[posIndex + 1] = y0 + (y1 - y0) * r;
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
			client.setConfig({ baseUrl: e.data.baseUrl, querySerializer });
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
