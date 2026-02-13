import { colord } from 'colord';

export function getDecorativeColors(baseColor: string) {
	const outlineColor = colord(baseColor).darken(0.2).toHex();
	const tinted = colord(baseColor).isDark()
		? colord(baseColor).lighten(0.35)
		: colord(baseColor).darken(0.35);
	const chevronColor = tinted.alpha(0.85).toRgbString();

	return { outlineColor, chevronColor };
}
