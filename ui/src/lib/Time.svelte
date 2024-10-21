<script lang="ts">
	import { formatTime } from './toDateTime';
	import { cn } from './utils';

	let {
		timestamp,
		delay,
		class: className,
		showAlways,
		rt,
		isRealtime
	}: {
		timestamp: string | undefined;
		delay: number | undefined;
		class?: string;
		showAlways?: boolean;
		rt: boolean;
		isRealtime: boolean;
	} = $props();

	const d = $derived(
		timestamp ? new Date(new Date(timestamp).getTime() - (rt ? (delay ?? 0) : 0)) : undefined
	);
	const highDelay = $derived(delay !== undefined ? delay >= 180000 : false);
</script>

<div
	class={cn('w-16', className)}
	class:text-destructive={isRealtime && rt && highDelay}
	class:text-green-600={isRealtime && rt && !highDelay}
>
	{#if d && (showAlways || !rt || (rt && isRealtime))}
		{formatTime(d)}
	{/if}
</div>
