<script lang="ts">
	import { Button, type ButtonProps } from '$lib/components/ui/button';
	import { formatDurationSec } from '$lib/formatDuration';
	import { getModeStyle, routeColor } from './modeStyle';
	import type { Itinerary } from '@motis-project/motis-client';

	const {
		d,
		...restProps
	}: {
		d: Itinerary;
	} & ButtonProps = $props();

	const modeStyles = [
		...new Map(d.legs.map((l) => [JSON.stringify(getModeStyle(l)), getModeStyle(l)])).values()
	];

	const leg = d.legs.find((leg) => leg.mode !== 'WALK') ?? d.legs[0]!;
</script>

<Button variant="child" {...restProps}>
	<div
		class="flex items-center py-1 px-2 rounded-lg font-bold text-sm h-8 text-nowrap"
		style={routeColor(leg)}
	>
		{#each modeStyles as [icon, _color, _textColor], i (i)}
			<svg class="relative mr-1 w-4 h-4 rounded-full">
				<use xlink:href={`#${icon}`}></use>
			</svg>
		{/each}
		{formatDurationSec(d.duration)}
	</div>
</Button>
