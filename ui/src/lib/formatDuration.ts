export const formatDurationMin = (t: number): string => {
	let hours = Math.floor(t / 60);
	let minutes = Math.ceil(t - hours * 60);
	if (minutes === 60) {
		hours += 1;
		minutes = 0;
	}
	const str = [
		hours !== 0 ? hours + ' h' : '',
		minutes !== 0 || hours === 0 ? minutes + ' min' : ''
	]
		.join(' ')
		.trim();
	return str;
};

export const formatDurationSec = (t: number): string => {
	let hours = Math.floor(t / 3600);
	let minutes = Math.ceil((t - hours * 3600) / 60);
	if (minutes === 60) {
		hours += 1;
		minutes = 0;
	}
	const str = [
		hours !== 0 ? hours + ' h' : '',
		minutes !== 0 || hours === 0 ? minutes + ' min' : ''
	]
		.join(' ')
		.trim();
	return str;
};

export const formatDistanceMeters = (m: number | undefined): string => {
	if (!m) return '';
	const kilometers = Math.floor(m / 1000);
	const meters = kilometers > 5 ? 0 : Math.ceil(m - kilometers * 1000);
	const str = [kilometers !== 0 ? kilometers + ' km' : '', meters !== 0 ? meters + ' m' : '']
		.join(' ')
		.trim();
	return str;
};
