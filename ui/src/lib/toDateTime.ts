import { language } from './i18n/translation';

export const formatTime = (d: Date, timeZone: string | undefined): string => {
	return d.toLocaleTimeString(language, {
		hour: 'numeric',
		minute: 'numeric',
		timeZone,
		hour12: false
	});
};

export const formatDate = (d: Date, timeZone: string | undefined): string => {
	return d.toLocaleDateString(language, {
		day: 'numeric',
		month: 'numeric',
		year: 'numeric',
		timeZone
	});
};
