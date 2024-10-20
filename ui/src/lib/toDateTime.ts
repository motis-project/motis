const pad = (x: number) => ('0' + x).slice(-2);

export const formatTime = (d: Date): string => {
	return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
};
