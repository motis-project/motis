import type { ExpressionSpecification } from 'maplibre-gl';

export const zoomScaledIconSize: ExpressionSpecification = [
	'interpolate',
	['linear'],
	['zoom'],
	14,
	0.6,
	18,
	1
];

export const zoomScaledTextSizeMedium: ExpressionSpecification = [
	'interpolate',
	['linear'],
	['zoom'],
	14,
	7.2,
	18,
	12
];

export const zoomScaledTextSizeSmall: ExpressionSpecification = [
	'interpolate',
	['linear'],
	['zoom'],
	14,
	6,
	18,
	10
];

export const createZoomScaledTextOffset = (
	baseOffset: [number, number]
): ExpressionSpecification => [
	'interpolate',
	['linear'],
	['zoom'],
	14,
	['literal', [baseOffset[0] * 0.8, baseOffset[1] * 0.9]],
	18,
	['literal', baseOffset]
];

export const zoomScaledTextOffset = createZoomScaledTextOffset([0.8, -1.25]);
