import { page } from '$app/state';
import { replaceState } from '$app/navigation';

export const restoreScroll = (container: HTMLElement) => {
	const saveScroll = () => {
		page.state.scrollY = container.scrollTop;
		replaceState('', page.state);
	};

	const handlePopState = (event: PopStateEvent) => {
		requestAnimationFrame(() => {
			container.scrollTop = event.state?.['sveltekit:states']?.scrollY ?? 0;
		});
	};

	container.addEventListener('scrollend', saveScroll);
	window.addEventListener('popstate', handlePopState);

	return () => {
		container.removeEventListener('scrollend', saveScroll);
		window.removeEventListener('popstate', handlePopState);
	};
};

export const resetScroll = (container: HTMLElement) => {
	if (page.state.scrollY == undefined) {
		container.scrollTop = 0;
	}
};
