import type { Leg } from '@motis-project/motis-client';

import { t } from './i18n/translation';

export const getModeName = (l: Leg) => {
	switch (l.mode) {
		case 'WALK':
			return t.walk;
		case 'BIKE':
			return t.bike;
		case 'RENTAL':
			switch (l.rental?.formFactor) {
				case 'BICYCLE':
					return t.bike;
				case 'CARGO_BICYCLE':
					return t.cargoBike;
				case 'CAR':
					return t.car;
				case 'MOPED':
					return t.moped;
				case 'SCOOTER_SEATED':
					return t.scooterSeated;
				case 'SCOOTER_STANDING':
					return t.scooterStanding;
				case 'OTHER':
				default:
					return t.unknownVehicleType;
			}
		case 'CAR':
		case 'CAR_PARKING':
			return t.car;
		case 'ODM':
			return t.taxi;
		case 'FLEX':
			return 'Flex';
		default:
			return `${l.mode}`;
	}
};
