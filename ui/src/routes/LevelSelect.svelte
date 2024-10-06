<script lang="ts">
	import { Label } from '$lib/components/ui/label';
	import { RadioGroup, Item } from '$lib/components/ui/radio-group/index';
	import { lngLatToStr } from '$lib/lngLatToStr';
	import Control from '$lib/map/Control.svelte';
	import { levels } from '$lib/openapi';
	import type { LngLatBoundsLike } from 'maplibre-gl';
	import maplibregl from 'maplibre-gl';

	let {
		bounds,
		zoom,
		level = $bindable()
	}: {
		bounds: LngLatBoundsLike | undefined;
		zoom: number | undefined;
		level: number;
	} = $props();

	let value = $state('level-0');
	let availableLevels: Array<string> = $state([]);

	$effect(() => {
		level = Number(value.substring('level-'.length));
	});

	$effect(() => {
		if (bounds && zoom && zoom > 17) {
			const b = maplibregl.LngLatBounds.convert(bounds);
			const min = lngLatToStr(b.getNorthWest());
			const max = lngLatToStr(b.getSouthEast());
			levels<false>({ query: { min, max } }).then((x) => {
				availableLevels =
					x.data
						?.filter((x) => {
							return Number.isInteger(x);
						})
						.map((x) => String(x)) ?? [];
			});
		} else {
			availableLevels = [];
		}
	});
</script>

{#if availableLevels.length > 1}
	<Control>
		<RadioGroup class="flex flex-col space-y-1" bind:value>
			{#each availableLevels as l}
				<Label
					for={`level-${l}`}
					class="inline-flex items-center justify-center font-bold rounded-md border-2 border-muted bg-popover h-9 w-9 hover:bg-accent hover:text-accent-foreground [&:has([data-state=checked])]:border-blue-600 hover:cursor-pointer"
				>
					<Item value={`level-${l}`} id={`level-${l}`} class="sr-only" aria-label={`level-${l}`} />
					{l}
				</Label>
			{/each}
		</RadioGroup>
	</Control>
{/if}
