const pad = (x: number) => ('0' + x).slice(-2);

export const toDateTime = (t: Date): [string, string] => {
	const date = `${pad(t.getUTCMonth() + 1)}-${pad(t.getUTCDate())}-${t.getUTCFullYear()}`;
	const time = `${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())}`;
	return [date, time];
};

export const formatTime = (d: Date): string => {
	return `${pad(d.getHours())}:${pad(d.getMinutes())}`
}