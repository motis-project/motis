import type { Feature, FeatureCollection, MultiPolygon } from 'geojson';

export type RentalZoneFeatureProperties = {
	zoneIndex: number;
	providerId: string;
	z: number;
	rideEndAllowed: boolean;
	rideThroughAllowed: boolean;
};

export type RentalZoneFeature = Feature<MultiPolygon, RentalZoneFeatureProperties>;

export type RentalZoneFeatureCollection = FeatureCollection<
	MultiPolygon,
	RentalZoneFeatureProperties
>;
