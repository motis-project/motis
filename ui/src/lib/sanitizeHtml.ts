import DOMPurify from 'dompurify';
import { browser } from '$app/environment';

// Service alerts (GTFS-RT / GTFS alerts) carry HTML in their text fields, e.g.
// "<b>Haltestellen:</b><ul><li>...</li></ul>". It comes from the feed, so it is
// untrusted input: everything outside this allow list is dropped, including all
// attributes except links.
const ALLOWED_TAGS = ['a', 'b', 'strong', 'i', 'em', 'u', 'br', 'p', 'span', 'ul', 'ol', 'li'];
const ALLOWED_ATTR = ['href', 'title'];

// DOMPurify needs a DOM. On the server it would silently pass the input through
// unchanged, so never emit markup there - alerts are only ever fetched in the
// browser anyway.
export const sanitizeAlertHtml = (html: string | undefined): string =>
	browser && html ? DOMPurify.sanitize(html, { ALLOWED_TAGS, ALLOWED_ATTR }) : '';

// Same input, but flattened to text - for previews and anywhere the markup
// would only get in the way. Block ends become spaces so that list items do not
// run into the text before them, and reading textContent decodes entities.
export const htmlToText = (html: string | undefined): string => {
	if (!browser || !html) {
		return html ?? '';
	}
	const el = document.createElement('div');
	el.innerHTML = DOMPurify.sanitize(html, { ALLOWED_TAGS, ALLOWED_ATTR }).replace(
		/<\/?(p|li|ul|ol|div|br)\b[^>]*>/gi,
		' '
	);
	return (el.textContent ?? '').replace(/\s+/g, ' ').trim();
};

if (browser) {
	// Links open in a new tab and must not be able to reach back into this one.
	DOMPurify.addHook('afterSanitizeAttributes', (node) => {
		if (node.tagName === 'A' && node.hasAttribute('href')) {
			node.setAttribute('target', '_blank');
			node.setAttribute('rel', 'noopener noreferrer');
		}
	});
}
