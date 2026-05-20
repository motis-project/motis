import { browser } from '$app/environment';

export function createTripIcon(size: number): HTMLCanvasElement | undefined {
	if (!browser) {
		return undefined;
	}

	const border = (2 / 64) * size;
	const padding = (size - size / 2) / 2 + border;
	const innerSize = size - 2 * padding;
	const mid = size / 2;
	const rad = innerSize / 3.5;
	const cv = document.createElement('canvas');
	cv.width = size;
	cv.height = size;
	const ctx = cv.getContext('2d', { alpha: true });

	if (!ctx) {
		return cv;
	}

	ctx.beginPath();

	ctx.arc(padding + rad, mid, rad, (1 / 2) * Math.PI, (3 / 2) * Math.PI, false);

	ctx.bezierCurveTo(padding + rad + rad, mid - rad, size - padding, mid, size - padding, mid);
	ctx.bezierCurveTo(size - padding, mid, padding + rad + rad, mid + rad, padding + rad, mid + rad);

	ctx.closePath();

	ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
	ctx.fill();
	ctx.lineWidth = border;
	ctx.strokeStyle = 'rgba(120, 120, 120, 1.0)';
	ctx.stroke();
	return cv;
}
