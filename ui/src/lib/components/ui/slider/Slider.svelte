<script lang="ts">
	import { Slider } from 'bits-ui';
	import cn from 'clsx';

	let {
		value = $bindable(),
		disabled,
		step,
		min,
		max,
		fill = true
	}: {
		value: number;
		disabled?: boolean;
		step?: number;
		min?: number;
		max?: number;
		/** Highlight the track between the minimum and the thumb. */
		fill?: boolean;
	} = $props();
</script>

<!--
  touch-none on the wrapper: bits-ui puts touch-action: pan-y on the root, so
  on a phone a slightly diagonal drag scrolls the page instead of the slider.
  The vertical padding widens the hit area, pointer down anywhere in the root
  grabs the thumb (the position is read from the root's left/right edges, so
  only vertical padding is safe).
-->
<div class="w-full touch-none {disabled ? 'opacity-10' : ''} md:max-w-[180px]">
	<Slider.Root
		type="single"
		{min}
		{max}
		{step}
		{disabled}
		bind:value
		class="relative flex w-full select-none items-center py-3"
	>
		<span class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-slate-200">
			{#if fill}
				<Slider.Range class="bg-foreground absolute h-full" />
			{/if}
		</span>
		<Slider.Thumb
			index={0}
			class={cn(
				'border-border-input bg-black border-black block size-5 cursor-pointer rounded-full shadow-sm'
			)}
		/>
	</Slider.Root>
</div>
