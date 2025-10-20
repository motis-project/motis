import { page } from '$app/state';
import { replaceState } from '$app/navigation';

export const restoreScroll = (container: HTMLElement) => {
	const saveScroll = () => {
		page.state.scrollY = container.scrollTop;
		replaceState('', page.state);
	};

	const handlePopState = () => {
		console.log(page.state);
		container.scrollTop = page.state.scrollY ?? 0;
	};

	container.addEventListener('scrollend', saveScroll);
	window.addEventListener('popstate', handlePopState);
};

export const resetScroll = (container: HTMLElement) => {
	if (page.state.selectedItinerary && page.state.scrollY == undefined) {
		container.scrollTop = 0;
	}
};
