<script lang="ts">
	import { cn } from './utils';

	let {
		value = $bindable(),
		now = false,
		class: className
	}: {
		value: Date;
		now?: boolean;
		class?: string;
	} = $props();

	let el: undefined | HTMLInputElement;

	$effect(() => {
		if (el !== undefined) {
			value.setSeconds(0, 0);
			// `now`: show the current minute instead of `value`, the input is disabled
			const date = now ? new Date(Math.floor(Date.now() / 60000) * 60000) : value;
			const dateTimeLocalValue = new Date(date.getTime() - date.getTimezoneOffset() * 60000)
				.toISOString()
				.slice(0, -1);
			el.value = dateTimeLocalValue;
		}
	});
</script>

<input
	type="datetime-local"
	class={cn(
		'flex h-9 rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50',
		className
	)}
	bind:this={el}
	disabled={now}
	onchange={(e) => {
		// @ts-expect-error target exists, value exists
		const dateTimeLocalValue = e.target!.value!;
		const fakeUtcTime = new Date(`${dateTimeLocalValue}Z`);
		if (!isNaN(fakeUtcTime.getTime())) {
			/* eslint-disable-next-line svelte/prefer-svelte-reactivity */
			value = new Date(fakeUtcTime.getTime() + fakeUtcTime.getTimezoneOffset() * 60000);
		}
	}}
/>
