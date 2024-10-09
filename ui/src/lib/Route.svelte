<script lang="ts">
	import { getModeStyle, routeColor, type LegLike } from './modeStyle';
	import { cn } from './utils';

	const { l, class: className }: { l: LegLike; class?: string } = $props();

	const modeIcon = getModeStyle(l.mode)[0];
</script>

<div
	class={cn(
		'w-fit flex items-center text-nowrap rounded-full pl-2 pr-1 h-8 py-[1px] font-bold',
		className,
		l.routeShortName ? 'min-w-28 pr-3' : undefined
	)}
	style={routeColor(l)}
>
	<svg class="relative mr-2 w-6 h-6 rounded-full">
		<use xlink:href={`#${modeIcon}`}></use>
	</svg>
	<div class="text-center">
		{l.routeShortName}
	</div>
</div>
