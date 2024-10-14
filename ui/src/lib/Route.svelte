<script lang="ts">
	import { getModeStyle, routeColor, type LegLike } from './modeStyle';
	import { cn } from './utils';

	const {
		l,
		class: className,
		onClickTrip
	}: {
		l: LegLike;
		class?: string;
		onClickTrip: (tripId: string, date: string) => void;
	} = $props();

	const modeIcon = getModeStyle(l.mode)[0];
</script>

<button
	class={cn(
		'flex items-center text-nowrap rounded-full pl-2 pr-1 h-8 font-bold',
		className,
		l.routeShortName ? 'pr-3' : undefined
	)}
	style={routeColor(l)}
	onclick={() => {
		if (l.tripId && l.serviceDate) {
			onClickTrip(l.tripId, l.serviceDate);
		} else {
			console.log('tripId or serviceDate missing', l);
		}
	}}
>
	<svg class="relative mr-2 min-w-6 min-h-6 max-w-6 max-h-6 rounded-full">
		<use xlink:href={`#${modeIcon}`}></use>
	</svg>
	<div class="text-center">
		{l.routeShortName}
	</div>
</button>
