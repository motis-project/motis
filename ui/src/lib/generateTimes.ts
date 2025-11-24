export const generateTimes = (limit: number): number[] => {
	const times: number[] = [];
	let t = 1;
	const defaultLimit = 6 * 60 * 60;
	const max = Math.min(defaultLimit, limit);
	while (t <= max / 60) {
		times.push(t * 60);
		if (t < 5) {
			t += 4;
		} else if (t < 30) {
			t += 5;
		} else if (t < 60) {
			t += 10;
		} else if (t < 120) {
			t += 30;
		} else {
			t += 60;
		}
	}
	if (times[times.length - 1] !== max) {
		times.push(max);
	}

	return times;
};
