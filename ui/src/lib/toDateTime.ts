import { language } from './i18n/translation';

const pad = (x: number) => ('0' + x).slice(-2);

export const formatTime = (d: Date): string => {
	return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

export const formatDate = (d: Date): string => {
	return d.toLocaleDateString(language, { day: 'numeric', month: 'numeric', year: 'numeric' });
};
