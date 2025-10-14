import type { RentalFormFactor } from '$lib/api/openapi';
import { t } from '$lib/i18n/translation';

export type FormFactorAssets = {
	svg: string;
	station: string;
	vehicle: string;
	cluster: string;
	label: string;
};

export const DEFAULT_FORM_FACTOR: RentalFormFactor = 'BICYCLE';

export const formFactorAssets: Record<RentalFormFactor, FormFactorAssets> = {
	BICYCLE: {
		svg: 'bike',
		station: 'bike_station',
		vehicle: 'floating_bike',
		cluster: 'floating_bike_cluster',
		label: t.RENTAL_BICYCLE
	},
	CARGO_BICYCLE: {
		svg: 'cargo_bike',
		station: 'cargo_bike_station',
		vehicle: 'floating_cargo_bike',
		cluster: 'floating_cargo_bike_cluster',
		label: t.RENTAL_CARGO_BICYCLE
	},
	CAR: {
		svg: 'car',
		station: 'car_station',
		vehicle: 'floating_car',
		cluster: 'floating_car_cluster',
		label: t.RENTAL_CAR
	},
	MOPED: {
		svg: 'moped',
		station: 'moped_station',
		vehicle: 'floating_moped',
		cluster: 'floating_moped_cluster',
		label: t.RENTAL_MOPED
	},
	SCOOTER_SEATED: {
		svg: 'scooter',
		station: 'scooter_station',
		vehicle: 'floating_scooter',
		cluster: 'floating_scooter_cluster',
		label: t.RENTAL_SCOOTER_SEATED
	},
	SCOOTER_STANDING: {
		svg: 'scooter',
		station: 'scooter_station',
		vehicle: 'floating_scooter',
		cluster: 'floating_scooter_cluster',
		label: t.RENTAL_SCOOTER_STANDING
	},
	OTHER: {
		svg: 'bike',
		station: 'bike_station',
		vehicle: 'floating_bike',
		cluster: 'floating_bike_cluster',
		label: t.RENTAL_OTHER
	}
};
