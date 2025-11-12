import { browser } from '$app/environment';
import type {
	RentalFormFactor,
	RentalPropulsionType,
	RentalReturnConstraint
} from '@motis-project/motis-client';
import {
	FlagTriangleLeft,
	Fuel,
	PlugZap,
	RefreshCcw,
	Zap,
	type Icon as LucideIcon
} from '@lucide/svelte';
import { t } from '$lib/i18n/translation';

export type FormFactorAssets = {
	svg: string;
	station: string;
	vehicle: string;
	cluster: string;
	label: string;
};

type IconDimensions = {
	width: number;
	height: number;
};

export type MapLibreImageSource = ImageBitmap | HTMLImageElement;

export const DEFAULT_FORM_FACTOR: RentalFormFactor = 'BICYCLE';

export const ICON_TYPES = ['station', 'vehicle', 'cluster'] as const;
export type IconType = (typeof ICON_TYPES)[number];

export const ICON_BASE_PATH = 'icons/rental/';

export const formFactorAssets: Record<RentalFormFactor, FormFactorAssets> = {
	BICYCLE: {
		svg: 'bike',
		station: 'bike_station',
		vehicle: 'floating_bike',
		cluster: 'floating_bike_cluster',
		label: t.bike
	},
	CARGO_BICYCLE: {
		svg: 'cargo_bike',
		station: 'cargo_bike_station',
		vehicle: 'floating_cargo_bike',
		cluster: 'floating_cargo_bike_cluster',
		label: t.cargoBike
	},
	CAR: {
		svg: 'car',
		station: 'car_station',
		vehicle: 'floating_car',
		cluster: 'floating_car_cluster',
		label: t.car
	},
	MOPED: {
		svg: 'moped',
		station: 'moped_station',
		vehicle: 'floating_moped',
		cluster: 'floating_moped_cluster',
		label: t.moped
	},
	SCOOTER_SEATED: {
		svg: 'seated_scooter',
		station: 'seated_scooter_station',
		vehicle: 'floating_seated_scooter',
		cluster: 'floating_seated_scooter_cluster',
		label: t.scooterSeated
	},
	SCOOTER_STANDING: {
		svg: 'scooter',
		station: 'scooter_station',
		vehicle: 'floating_scooter',
		cluster: 'floating_scooter_cluster',
		label: t.scooterStanding
	},
	OTHER: {
		svg: 'other',
		station: 'other_station',
		vehicle: 'floating_other',
		cluster: 'floating_other_cluster',
		label: t.unknownVehicleType
	}
};

export const propulsionTypes: Record<
	RentalPropulsionType,
	{ component: typeof LucideIcon; title: string } | null
> = {
	ELECTRIC: { component: Zap, title: t.electric },
	ELECTRIC_ASSIST: { component: Zap, title: t.electricAssist },
	HYBRID: { component: PlugZap, title: t.hybrid },
	PLUG_IN_HYBRID: { component: PlugZap, title: t.plugInHybrid },
	COMBUSTION: { component: Fuel, title: t.combustion },
	COMBUSTION_DIESEL: { component: Fuel, title: t.combustionDiesel },
	HYDROGEN_FUEL_CELL: { component: Fuel, title: t.hydrogenFuelCell },
	HUMAN: null
};

export const returnConstraints: Record<
	RentalReturnConstraint,
	{ component: typeof LucideIcon; title: string } | null
> = {
	ANY_STATION: { component: FlagTriangleLeft, title: t.returnOnlyAtStations },
	ROUNDTRIP_STATION: { component: RefreshCcw, title: t.roundtripStationReturnConstraint },
	NONE: null
};

export const getIconBaseName = (formFactor: RentalFormFactor, type: IconType) =>
	formFactorAssets[formFactor][type];

export const getIconUrl = (formFactor: RentalFormFactor, type: IconType) =>
	`${ICON_BASE_PATH}${getIconBaseName(formFactor, type)}.svg`;

const iconTypeDimensions: Record<IconType, IconDimensions> = {
	station: { width: 43, height: 44 },
	vehicle: { width: 27, height: 27 },
	cluster: { width: 35, height: 36 }
};

export const getIconDimensions = (type: IconType): IconDimensions => iconTypeDimensions[type];

export async function colorizeIcon(
	svgUrl: string,
	color: string,
	dimensions: IconDimensions
): Promise<MapLibreImageSource> {
	if (!browser) {
		throw new Error('colorizeIcon is not supported in this environment');
	}

	const response = await fetch(svgUrl);
	if (!response.ok) {
		throw new Error(`Failed to load icon: ${response.status} ${response.statusText}`);
	}

	const svgContent = await response.text();
	const parser = new DOMParser();
	const svgDoc = parser.parseFromString(svgContent, 'image/svg+xml');

	if (svgDoc.getElementsByTagName('parsererror').length > 0) {
		throw new Error('Invalid SVG content');
	}

	const rootElement = svgDoc.documentElement;
	if (!(rootElement instanceof SVGSVGElement)) {
		throw new Error('Provided file is not an SVG');
	}
	const svgRoot = rootElement;

	if (!svgRoot.getAttribute('xmlns')) {
		svgRoot.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
	}

	const existingStyle = svgRoot.getAttribute('style');
	const colorStyle = `color: ${color}`;
	const mergedStyle = existingStyle ? `${existingStyle};${colorStyle}` : colorStyle;
	svgRoot.setAttribute('style', mergedStyle);
	svgRoot.setAttribute('color', color);
	svgRoot.setAttribute('width', `${dimensions.width}`);
	svgRoot.setAttribute('height', `${dimensions.height}`);

	const serializer = new XMLSerializer();
	const serialized = serializer.serializeToString(svgDoc);
	const blob = new Blob([serialized], { type: 'image/svg+xml;charset=utf-8' });

	if (typeof createImageBitmap === 'function') {
		try {
			const bitmap = await createImageBitmap(blob);
			return bitmap;
		} catch (_error) {} // eslint-disable-line
	}

	return await new Promise<MapLibreImageSource>((resolve, reject) => {
		const blob_url = URL.createObjectURL(blob);
		const image = new Image();
		image.crossOrigin = 'anonymous';
		image.decoding = 'async';
		image.onload = () => {
			URL.revokeObjectURL(blob_url);
			if (dimensions) {
				image.width = dimensions.width;
				image.height = dimensions.height;
			}
			resolve(image);
		};
		image.onerror = () => {
			URL.revokeObjectURL(blob_url);
			reject(new Error('Failed to load generated image'));
		};
		image.src = blob_url;
	});
}
