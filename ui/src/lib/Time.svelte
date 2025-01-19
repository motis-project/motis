<script lang="ts">
	import { formatTime } from './toDateTime';
	import { cn } from './utils';

	let {
		class: className,
		timestamp,
		scheduledTimestamp,
		isRealtime,
		variant
	}: {
		class?: string;
		timestamp: string;
		scheduledTimestamp: string;
		isRealtime: boolean;
		variant: 'schedule' | 'realtime' | 'realtime-show-always';
	} = $props();

	const t = $derived(new Date(timestamp));
	const scheduled = $derived(new Date(scheduledTimestamp));
	const delayMinutes = $derived((t.getTime() - scheduled.getTime()) / 60000);
	const highDelay = $derived(isRealtime && delayMinutes > 3);
	const lowDelay = $derived(isRealtime && delayMinutes <= 3);
</script>

<div class={cn('text-nowrap', className)}>
	{#if variant == 'schedule'}
		{formatTime(scheduled)}
	{:else if variant === 'realtime-show-always' || (variant === 'realtime' && isRealtime)}
		<div class:text-destructive={highDelay} class:text-green-600={lowDelay}>
			{formatTime(t)}
		</div>
	{/if}
</div>
