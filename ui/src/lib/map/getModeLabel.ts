import type { Mode } from '@motis-project/motis-client';

export const getModeLabel = (mode: Mode): string => {
	switch (mode) {
		case 'BUS':
		case 'FERRY':
		case 'TRAM':
		case 'COACH':
		case 'AIRPLANE':
		case 'AERIAL_LIFT':
			return 'Platform';
		default:
			return 'Track';
	}
};
