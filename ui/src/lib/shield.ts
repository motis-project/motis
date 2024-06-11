import { browser } from '$app/environment';

class ShieldOptions {
	fill!: string;
	stroke!: string;
}

export function createShield(opt: ShieldOptions): [ImageData, Object] {
	if (!browser) {
		throw 'not supported';
	}

	const d = 32;

	const cv = document.createElement('canvas');
	cv.width = d;
	cv.height = d;
	const ctx = cv.getContext('2d')!;

	// coord of the line (front = near zero, back = opposite)
	const l_front = 1;
	const l_back = d - 1;

	// coord start of the arc
	const lr_front = l_front + 2;
	const lr_back = l_back - 2;

	// control point of the arc
	const lp_front = l_front + 1;
	const lp_back = l_back - 1;

	let p = new Path2D();
	p.moveTo(lr_front, l_front);

	// top line
	p.lineTo(lr_back, l_front);
	// top right corner
	p.bezierCurveTo(lp_back, lp_front, lp_back, lp_front, l_back, lr_front);
	// right line
	p.lineTo(l_back, lr_back);
	// bottom right corner
	p.bezierCurveTo(lp_back, lp_back, lp_back, lp_back, lr_back, l_back);
	// bottom line
	p.lineTo(lr_front, l_back);
	// bottom left corner
	p.bezierCurveTo(lp_front, lp_back, lp_front, lp_back, l_front, lr_back);
	// left line
	p.lineTo(l_front, lr_front);
	// top left corner
	p.bezierCurveTo(lp_front, lp_front, lp_front, lp_front, lr_front, l_front);

	p.closePath();

	ctx.fillStyle = opt.fill;
	ctx.fill(p);
	ctx.strokeStyle = opt.stroke;
	ctx.stroke(p);

	return [
		ctx.getImageData(0, 0, d, d),
		{
			content: [lr_front, lr_front, lr_back, lr_back],
			stretchX: [[lr_front, lr_back]],
			stretchY: [[lr_front, lr_back]]
		}
	];
}
