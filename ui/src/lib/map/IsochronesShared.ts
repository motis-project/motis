export type StatusLevel = 'WORKING' | 'DONE' | 'EMPTY' | 'FAILED';

export interface IsochronesOptions {
	color: string;
	opacity: number;
	status: StatusLevel;
	errorMessage: string | undefined;
	errorCode: number | undefined;
}
export interface IsochronesPos {
	lat: number;
	lng: number;
	seconds: number;
}
