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

<div class="w-full {disabled ? 'opacity-10' : ''} md:max-w-[180px]">
	<Slider.Root
		type="single"
		{min}
		{max}
		{step}
		{disabled}
		bind:value
		class="relative bg-slate-200 flex rounded w-full touch-none select-none items-center"
	>
		<span class="bg-dark-10 relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full">
			{#if fill}
				<Slider.Range class="bg-foreground absolute h-full" />
			{/if}
		</span>
		<Slider.Thumb
			index={0}
			class={cn(
				'border-border-input bg-black border-black block size-[15px] cursor-pointer rounded-full shadow-sm'
			)}
		/>
	</Slider.Root>
</div>
