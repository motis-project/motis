import type { Match } from './openapi';

export type Location = {
	label?: string;
	value: {
		match?: Match;
		precision?: number;
		level?: number;
	};
};
