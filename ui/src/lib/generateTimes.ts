export const generateTimes = (limit: number | undefined, defaultLimit: number): number[] => {
	const times: number[] = [];
	let t = 1;
	const max = limit ?? defaultLimit;
	while (t <= max / 60) {
		times.push(t * 60);
		if (t < 30) {
			t += 5;
		} else if (t < 60) {
			t += 10;
		} else if (t < 120) {
			t += 30;
		} else {
			t += 60;
		}
	}
	return times;
};
