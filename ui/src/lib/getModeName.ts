import type { Mode } from './openapi';

export const getModeName = (m: Mode) => {
	switch (m) {
		case 'WALK':
			return 'Fußweg';
		case 'BIKE':
		case 'BIKE_RENTAL':
		case 'BIKE_TO_PARK':
			return 'Fahrrad';
    case 'SCOOTER_RENTAL':
      return 'Scooter';
		case 'CAR':
		case 'CAR_TO_PARK':
		case 'CAR_HAILING':
		case 'CAR_PICKUP':
		case 'CAR_RENTAL':
		case 'CAR_SHARING':
			return 'Auto';
		default:
			return `${m}`;
	}
};
