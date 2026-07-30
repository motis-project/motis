export type StatusLevel = 'WORKING' | 'DONE' | 'EMPTY' | 'FAILED';

export interface IsochronesOptions {
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
