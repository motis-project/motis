<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { Slider } from 'bits-ui';

	let {
		value = $bindable(),
		color = $bindable()
	}: {
		value: number;
		color: string;
	} = $props();

</script>

<div class="grid grid-cols-[1fr_2fr_1fr] items-center gap-2">
	<div class="text-sm">
		Style
	</div>
	<Slider.Root
		type="single"
		min={0}
		max={1000}
		bind:value
		class="relative flex w-full touch-none select-none items-center"
	>
		{#snippet children()}
			<span
				class="bg-dark-10 relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full"
			>
				<Slider.Range class="bg-foreground absolute h-full" />
			</span>
			<Slider.Thumb
				index={0}
				class="border-border-input bg-background hover:border-dark-40 focus-visible:ring-foreground dark:bg-foreground dark:shadow-card focus-visible:outline-hidden block size-[25px] cursor-pointer rounded-full border shadow-sm transition-colors focus-visible:ring-2 focus-visible:ring-offset-2 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50"
			/>
		{/snippet}
	</Slider.Root>
	<input class="flex right-0 align-right" type="color" bind:value={color} />
</div>
