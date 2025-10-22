import type { ExpressionSpecification } from 'maplibre-gl';

export const createZoomScaledSize = (baseSize: number): ExpressionSpecification => [
	'interpolate',
	['linear'],
	['zoom'],
	14,
	baseSize * 0.6,
	18,
	baseSize
];

export const zoomScaledIconSize = createZoomScaledSize(1);
export const zoomScaledTextSizeMedium = createZoomScaledSize(12);
export const zoomScaledTextSizeSmall = createZoomScaledSize(10);

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

export const DEFAULT_COLOR = '#2563eb';
export const DEFAULT_CONTRAST_COLOR = '#ffffff';
