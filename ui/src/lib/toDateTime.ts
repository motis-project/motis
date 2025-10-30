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

export const formatDateTime = (d: Date, timeZone: string | undefined): string => {
	return d.toLocaleDateString(language, {
		day: 'numeric',
		month: 'numeric',
		year: 'numeric',
		hour: 'numeric',
		minute: 'numeric',
		timeZone
	});
};

export const getTz = (d: Date, timeZone: string | undefined): string | undefined => {
	const timeZoneOffset = new Intl.DateTimeFormat(language, {
		timeZone,
		timeZoneName: 'shortOffset'
	})
		.formatToParts(d)
		.find((part) => part.type === 'timeZoneName')!.value;
	const isSameAsBrowserTimezone =
		new Intl.DateTimeFormat(language, { timeZoneName: 'shortOffset' })
			.formatToParts(d)
			.find((part) => part.type === 'timeZoneName')!.value == timeZoneOffset;
	return isSameAsBrowserTimezone ? undefined : timeZoneOffset;
};
