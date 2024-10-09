<script lang="ts">
	import { cn } from './utils';

	let {
		timestamp,
		delay,
		class: className,
		showAlways,
		rt,
		isRealtime
	}: {
		timestamp: number;
		delay: number;
		class?: string;
		showAlways?: boolean;
		rt: boolean;
		isRealtime: boolean;
	} = $props();

	const d = new Date(rt ? timestamp : timestamp - delay);
	const pad = (x: number) => ('0' + x).slice(-2);
</script>

<div
	class={cn('w-16', className)}
	class:text-destructive={isRealtime && rt && delay >= 180000}
	class:text-green-600={isRealtime && rt && delay < 180000}
>
	{#if showAlways || !rt || (rt && isRealtime)}
		{pad(d.getHours())}:{pad(d.getMinutes())}
	{/if}
</div>
