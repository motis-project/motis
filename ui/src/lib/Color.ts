export type RGBA = [number, number, number, number];
export function hexToRgb(hex: string): RGBA {
	const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	if (!result) {
		throw `${hex} is not a hex color #RRGGBB`;
	}
	return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16), 255];
}

export function rgbToHex(rgba: RGBA): string {
	return '#' + ((1 << 24) | (rgba[0] << 16) | (rgba[1] << 8) | rgba[2]).toString(16).slice(1);
}

export const getDelayColor = (delay: number, realTime: boolean): RGBA => {
	delay = delay / 60000;
	if (!realTime) {
		return [100, 100, 100, 255];
	}
	if (delay <= -30) {
		return [255, 0, 255, 255];
	} else if (delay <= -6) {
		return [138, 82, 254, 255];
	} else if (delay <= 3) {
		return [69, 194, 74, 255];
	} else if (delay <= 5) {
		return [255, 237, 0, 255];
	} else if (delay <= 10) {
		return [255, 102, 0, 255];
	} else if (delay <= 15) {
		return [255, 48, 71, 255];
	}
	return [163, 0, 10, 255];
};
