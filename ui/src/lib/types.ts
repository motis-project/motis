import type { Mode } from '@motis-project/motis-client';

export type Position = [number, number];
export type Trip = {
	realtime: boolean;
	routeColor?: string;
	mode: Mode;
	departureDelay: number;
	path: Float64Array;
	arrivalDelay: number;
	timestamps: Float64Array;
	currentIndx: number;
	headings: Float32Array;
};
export type TransferData = {
	length: number;
	positions: Float64Array;
	angles: Float32Array;
	colors: Uint8Array;
};
export type MetaData = {
	id: string;
	displayName?: string;
	tz?: string;
	from: string;
	to: string;
	realtime: boolean;
	arrival: string;
	departure: string;
	scheduledArrival: string;
	scheduledDeparture: string;
	departureDelay: number;
	arrivalDelay: number;
};
export type Query = {
	min: string;
	max: string;
	startTime: string;
	endTime: string;
	zoom: number;
};
