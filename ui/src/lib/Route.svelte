<script lang="ts">
	import { getModeStyle, routeColor, type LegLike } from './modeStyle';
	import { cn } from './utils';

	const { l, class: className }: { l: LegLike; class?: string } = $props();

	const modeIcon = getModeStyle(l.mode)[0];
</script>

<div
	class={cn(
		'flex items-center text-nowrap rounded-full pl-2 pr-3 h-8 py-[1px] font-bold min-w-28',
		className
	)}
	style={routeColor(l)}
>
	<svg class="relative mr-2 w-6 h-6 fill-white rounded-full">
		<use xlink:href={`#${modeIcon}`}></use>
	</svg>
	<div class="flex-grow text-center">{l.routeShortName}</div>
</div>
