<script lang="ts">
	import { language } from '$lib/i18n/translation';
	import { formatTime } from './toDateTime';
	import { cn } from './utils';

	let {
		class: className,
		timestamp,
		scheduledTimestamp,
		isRealtime,
		variant,
		queriedTime,
		timeZone
	}: {
		class?: string;
		timestamp: string;
		scheduledTimestamp: string;
		isRealtime: boolean;
		variant: 'schedule' | 'realtime' | 'realtime-show-always';
		queriedTime?: string | undefined;
		timeZone: string | undefined;
	} = $props();

	const t = $derived(new Date(timestamp));
	const scheduled = $derived(new Date(scheduledTimestamp));
	const delayMinutes = $derived((t.getTime() - scheduled.getTime()) / 60000);
	const highDelay = $derived(isRealtime && delayMinutes > 3);
	const lowDelay = $derived(isRealtime && delayMinutes <= 3);

	function weekday(time: Date) {
		if (variant === 'realtime') {
			return '';
		}
		if (queriedTime === undefined) {
			return time.toLocaleDateString(language, { timeZone });
		}
		const base = new Date(queriedTime);
		return base.toLocaleDateString() === time.toLocaleDateString()
			? ''
			: `(${time.toLocaleString(language, { weekday: 'short', timeZone })})`;
	}
</script>

<div class={cn('text-nowrap', className)}>
	{#if variant == 'schedule'}
		{formatTime(scheduled, timeZone)}
		{weekday(scheduled)}
	{:else if variant === 'realtime-show-always' || (variant === 'realtime' && isRealtime)}
		<span class:text-destructive={highDelay} class:text-green-600={lowDelay}>
			{formatTime(t, timeZone)}
		</span>
		{weekday(t)}
	{/if}
</div>
