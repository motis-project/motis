<script lang="ts">
	import { cn } from './utils';

	let {
		timestamp,
		delay,
		class: className,
		showAlways
	}: {
		timestamp: number;
		delay?: number;
		class?: string;
		showAlways?: boolean;
	} = $props();

	const d = new Date(timestamp + (delay || 0));
	const pad = (x: number) => ('0' + x).slice(-2);
</script>

<div
	class={cn('w-16', className)}
	class:text-destructive={delay && delay >= 180000}
	class:text-green-600={delay && delay < 180000}
>
	{#if showAlways || delay === undefined || (delay !== undefined && delay !== 0)}
		{pad(d.getHours())}:{pad(d.getMinutes())}
	{/if}
</div>
