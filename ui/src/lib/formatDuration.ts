export const formatDurationSec = (t: number) => {
	const hours = Math.floor(t / 3600);
	const minutes = Math.floor((t - hours * 3600) / 60);
	const seconds = t % 60;
	const str = [
		hours !== 0 ? hours + ' h' : '',
		minutes !== 0 ? minutes + ' min' : '',
		seconds !== 0 ? seconds + ' s' : ''
	]
		.join(' ')
		.trim();
	return str;
};
