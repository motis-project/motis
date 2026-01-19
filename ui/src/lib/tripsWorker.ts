/// <reference lib="webworker" />trips
import polyline from '@mapbox/polyline';
import { client, trips, type TripSegment } from '@motis-project/motis-client';
import type { Trip, Position, TransferData, MetaData, Query } from './types';
import type { QuerySerializerOptions } from '@hey-api/client-fetch';
import { getDelayColor, hexToRgb } from './Color';
import { getModeStyle, getColor } from './modeStyle';

// MATH
const toRad = (deg: number) => (deg * Math.PI) / 180;
const toDeg = (rad: number) => (rad * 180) / Math.PI;
const getSpatialData = (p0: Position, p1: Position) => {
	const [lon1, lat1] = p0.map(toRad);
	const [lon2, lat2] = p1.map(toRad);
	// Haversine Distance
	const R = 6371; // Earth radius in km
	const dLat = lat2 - lat1;
	const dLon = lon2 - lon1;
	const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
	const dist = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)) * R;

	// Spherical Bearing
	const y = Math.sin(dLon) * Math.cos(lat2);
	const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
	const brng = (toDeg(Math.atan2(y, x)) + 360) % 360;

	return { dist, brng };
};
const ease = (t: number) => t * t * (3 - 2 * t);
const lerp = (p0: Position, p1: Position, t0: number, t1: number, time: number): Position => {
	const r = ease((time - t0) / (t1 - t0));
	return [p0[0] + (p1[0] - p0[0]) * r, p0[1] + (p1[1] - p0[1]) * r];
};

//PROCESSING
const metadata: MetaData[] = [];
let index = 0;
let status: number;
let tripsData: Trip[] = [];
const fetchData = async (query: Query) => {
	const { data, response } = await trips({ query });
	status = response.status;
	if (!data) return;
	index = 0;
	tripsData = data.map((t) => processSegment(t));
};
const processSegment = (s: TripSegment): Trip => {
	const departure = new Date(s.departure).getTime();
	const arrival = new Date(s.arrival).getTime();
	const scheduledArrival = new Date(s.scheduledArrival).getTime();
	const scheduledDeparture = new Date(s.scheduledDeparture).getTime();
	const totalDuration = arrival - departure;
	const arrivalDelay = arrival - scheduledArrival;
	const departureDelay = departure - scheduledDeparture;
	const decoded = polyline.decode(s.polyline) as Position[];
	const count = decoded.length;
	const path: Position[] = new Array(count);
	const timestamps = new Float64Array(count);
	const headings = new Float32Array(count);
	const segmentDistances = new Float64Array(count - 1);
	let totalDistance = 0;

	for (let i = 0; i < count - 1; i++) {
		path[i] = [decoded[i][1], decoded[i][0]];
		path[i + 1] = [decoded[i + 1][1], decoded[i + 1][0]];
		const { dist, brng } = getSpatialData(path[i], path[i + 1]);
		segmentDistances[i] = dist;
		headings[i] = -brng + 90;
		totalDistance += dist;
	}
	headings[count - 1] = headings[count - 2];

	let cumulativeDist = 0;
	for (let i = 0; i < count; i++) {
		const progress = totalDistance === 0 ? 0 : cumulativeDist / totalDistance;
		timestamps[i] = departure + progress * totalDuration;

		if (i < count - 1) {
			cumulativeDist += segmentDistances[i];
		}
	}
	metadata[index] = {
		id: s.trips[0].tripId,
		displayName: s.trips[0].displayName,
		tz: s.from.tz,
		from: s.from.name,
		to: s.to.name,
		realtime: s.realTime,
		arrival: s.arrival,
		departure: s.departure,
		scheduledArrival: s.scheduledArrival,
		scheduledDeparture: s.scheduledArrival,
		departureDelay,
		arrivalDelay
	};
	index++;
	return {
		realtime: s.realTime,
		mode: s.mode,
		routeColor: s.routeColor,
		path,
		timestamps,
		headings,
		currentIndx: 0,
		departureDelay,
		arrivalDelay
	};
};

//STATE UPDATE
function updateState(data: TransferData, colorMode: string) {
	let posIndex = 0;
	let colorIndex = 0;
	let angleIndex = 0;
	const time = Date.now();
	for (const d of tripsData) {
		let color;
		switch (colorMode) {
			case 'rt':
				color = getDelayColor(d.departureDelay, d.realtime);
				break;
			case 'mode':
				color = hexToRgb(getModeStyle(d)[1]);
				break;
			case 'route':
				color = hexToRgb(getColor(d)[0]);
				break;
			case 'none':
				color = hexToRgb(getColor(d)[0]);
				break;
		}
		data.colors[colorIndex] = color![0];
		data.colors[colorIndex + 1] = color![1];
		data.colors[colorIndex + 2] = color![2];

		const len = d.path.length;
		const last = d.timestamps[len - 1];
		while (d.currentIndx < len - 1 && d.timestamps[d.currentIndx] < time) {
			d.currentIndx++;
		}
		if (d.currentIndx === 0) {
			data.positions[posIndex] = 0;
			data.positions[posIndex + 1] = 0;
			data.angles[angleIndex] = d.headings[0];
		} else if (d.currentIndx === len - 1 && time >= last) {
			data.positions[posIndex] = 0;
			data.positions[posIndex + 1] = 0;
			data.angles[angleIndex] = d.headings[len - 1];
		} else {
			const pos = lerp(
				d.path[d.currentIndx - 1],
				d.path[d.currentIndx],
				d.timestamps[d.currentIndx - 1],
				d.timestamps[d.currentIndx],
				time
			);
			data.positions[posIndex] = pos[0];
			data.positions[posIndex + 1] = pos[1];
			data.angles[angleIndex] = d.headings[d.currentIndx - 1];
		}
		colorIndex += 3;
		angleIndex++;
		posIndex += 2;
	}
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
			postMessage({ type: 'fetch-complete', status, metadata });
			break;
		case 'update': {
			const positions = new Float64Array(e.data.positions.buffer);
			const angles = new Float32Array(e.data.angles.buffer);
			const colors = new Uint8Array(e.data.colors.buffer);
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
					length: data.length
				},
				[data.angles.buffer, data.positions.buffer, data.colors.buffer]
			);
			break;
		}
	}
};

export {};
