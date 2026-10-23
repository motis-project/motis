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

export const formatLongDate = (d: Date, timeZone: string | undefined): string => {
	return d.toLocaleDateString(language, {
		weekday: 'long',
		day: 'numeric',
		month: 'long',
		year: d.getFullYear() === new Date().getFullYear() ? undefined : 'numeric',
		timeZone
	});
};

// calendar day of d in timeZone; any fixed locale works, en-CA just yields readable YYYY-MM-DD keys
const dayKey = (d: Date, timeZone: string | undefined): string =>
	d.toLocaleDateString('en-CA', { timeZone });

export const isSameDay = (a: Date, b: Date, timeZone: string | undefined): boolean =>
	dayKey(a, timeZone) === dayKey(b, timeZone);

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

export const formatDateTimeWithTimeZone = (d: Date, timeZone: string | undefined): string => {
	let output = formatDateTime(d, timeZone);
	const timeZoneOffset = getTz(d, timeZone);
	if (timeZoneOffset) {
		output += ' ' + timeZoneOffset;
	}
	return output;
};
