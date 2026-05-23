const normalize = (str: string) => str.replace(/[()[\]]/g, '');

export const normalizedContains = (haystack: string, needle: string): boolean => {
	return normalize(haystack).includes(normalize(needle));
};
