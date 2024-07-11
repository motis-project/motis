export const formatDurationSec = (t: number) => {
	const hours = Math.floor(t / 3600);
	const minutes = (t - hours * 3600) / 60;
	const str = [hours !== 0 ? hours + ' h' : '', minutes !== 0 ? minutes + ' min' : '']
		.join(' ')
		.trim();
	return str;
};
