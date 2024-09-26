<script lang="ts">
	import { cn } from './utils';

	let {
		timestamp,
		delay,
		class: className
	}: {
		timestamp: number;
		delay?: number;
		class?: string;
	} = $props();

	const d = new Date(timestamp + (delay || 0));
	const pad = (x: number) => ('0' + x).slice(-2);
</script>

<div class={cn('w-16', className)}>
	{#if delay === undefined || (delay !== undefined && delay !== 0)}
		{pad(d.getHours())}:{pad(d.getMinutes())}
	{/if}
</div>
