import { browser } from '$app/environment';
import type maplibregl from 'maplibre-gl';
import type { Mode } from '@motis-project/motis-client';
import { getModeStyle, type LegLike } from '$lib/modeStyle';

const MARKER_SIZE = 24;
const PIXEL_RATIO = 2;

// Attributes that may carry presentation info on the <symbol> elements in
// app.html. They are copied onto the standalone <svg> so `currentColor`
// resolves and stroke-based icons (taxi, aerial_lift, ...) keep their stroke.
const PRESENTATION_ATTRS = [
	'fill',
	'stroke',
	'stroke-width',
	'stroke-linecap',
	'stroke-linejoin',
	'stroke-miterlimit',
	'fill-rule',
	'clip-rule',
	'style'
] as const;

/** maplibre image id for a stop marker of the given mode. */
export const stopIconId = (mode: Mode | undefined): string => `stop-${mode ?? 'OTHER'}`;

/**
 * Builds a standalone, single-color SVG from one of the inline <symbol>
 * definitions in app.html and rasterizes it into an <img>.
 */
function renderGlyph(symbolId: string, color: string, size: number): Promise<HTMLImageElement> {
	const symbol = document.getElementById(symbolId);
	if (!(symbol instanceof SVGSymbolElement)) {
		return Promise.reject(new Error(`SVG symbol #${symbolId} not found in app.html`));
	}

	const viewBox = symbol.getAttribute('viewBox') ?? '0 0 24 24';
	const attrs = PRESENTATION_ATTRS.flatMap((a) => {
		const v = symbol.getAttribute(a);
		return v == null ? [] : [`${a}="${v.replace(/"/g, '&quot;')}"`];
	});
	// Icons that specify neither fill nor stroke inherit a fill from the root.
	if (!symbol.hasAttribute('fill') && !symbol.hasAttribute('stroke')) {
		attrs.push(`fill="${color}"`);
	}

	const svg =
		`<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox}" ` +
		`width="${size}" height="${size}" color="${color}" ${attrs.join(' ')}>` +
		`${symbol.innerHTML}</svg>`;

	const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }));
	return new Promise<HTMLImageElement>((resolve, reject) => {
		const image = new Image();
		image.decoding = 'async';
		image.onload = () => {
			URL.revokeObjectURL(url);
			resolve(image);
		};
		image.onerror = () => {
			URL.revokeObjectURL(url);
			reject(new Error(`Failed to render SVG symbol #${symbolId}`));
		};
		image.src = url;
	});
}

/** Renders a colored circle marker with the mode's SVG glyph centered on it. */
async function createStopMarkerImage(mode: Mode): Promise<ImageData> {
	const [symbolId, bg, fg] = getModeStyle({ mode } as LegLike);

	const size = MARKER_SIZE * PIXEL_RATIO;
	const canvas = document.createElement('canvas');
	canvas.width = size;
	canvas.height = size;
	const ctx = canvas.getContext('2d')!;

	const center = size / 2;
	const border = 1.5 * PIXEL_RATIO;
	const radius = center - border;

	ctx.beginPath();
	ctx.arc(center, center, radius, 0, Math.PI * 2);
	ctx.fillStyle = bg;
	ctx.fill();
	ctx.lineWidth = border;
	ctx.strokeStyle = '#ffffff';
	ctx.stroke();

	const glyphPx = Math.round(radius * 1.25);
	const glyph = await renderGlyph(symbolId, fg, glyphPx);
	ctx.drawImage(glyph, center - glyphPx / 2, center - glyphPx / 2, glyphPx, glyphPx);

	return ctx.getImageData(0, 0, size, size);
}

/**
 * Registers stop marker images for every given mode (plus the OTHER fallback)
 * on the map, skipping any that already exist. Safe to call repeatedly.
 */
export async function ensureStopIcons(map: maplibregl.Map, modes: Mode[]): Promise<void> {
	if (!browser) {
		return;
	}
	const wanted = new Set<Mode>(modes);
	wanted.add('OTHER');
	await Promise.all(
		[...wanted].map(async (mode) => {
			const id = stopIconId(mode);
			if (map.hasImage(id)) {
				return;
			}
			const image = await createStopMarkerImage(mode);
			if (!map.hasImage(id)) {
				map.addImage(id, image, { pixelRatio: PIXEL_RATIO });
			}
		})
	);
}
