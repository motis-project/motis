<script lang="ts">
	import { getModeStyle, routeColor, type LegLike } from './modeStyle';
	import { cn } from './utils';

	const {
		l,
		class: className,
		onClickTrip,
		compact = false
	}: {
		l: LegLike;
		class?: string;
		onClickTrip: (tripId: string) => void;
		compact?: boolean;
	} = $props();

	const modeIcon = $derived(getModeStyle(l)[0]);
</script>

<button
	class={cn(
		'flex items-center text-nowrap rounded-full font-bold',
		compact ? 'pl-1.5 pr-0.5 h-6 text-xs' : 'pl-2 pr-1 h-8',
		l.displayName ? (compact ? 'pr-2' : 'pr-3') : undefined,
		className
	)}
	style={routeColor(l)}
	onclick={() => {
		if (l.tripId) {
			onClickTrip(l.tripId);
		} else {
			console.log('tripId missing', l);
		}
	}}
>
	<svg
		class={cn(
			'relative',
			compact ? 'mr-1 min-w-4 min-h-4 max-w-4 max-h-4' : 'mr-2 min-w-6 min-h-6 max-w-6 max-h-6'
		)}
	>
		<use xlink:href={`#${modeIcon}`}></use>
	</svg>
	<div class="text-center">
		{l.displayName}
	</div>
</button>
