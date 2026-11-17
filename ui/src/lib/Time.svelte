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
		timeZone,
		arriveBy
	}: {
		class?: string;
		timestamp: string;
		scheduledTimestamp: string;
		isRealtime: boolean;
		variant: 'schedule' | 'realtime' | 'realtime-show-always';
		timeZone: string | undefined;
		arriveBy?: boolean | undefined;
	} = $props();

	const t = $derived(new Date(timestamp));
	const scheduled = $derived(new Date(scheduledTimestamp));
	const delayMinutes = $derived((t.getTime() - scheduled.getTime()) / 60000);
	const highDelay = $derived(isRealtime && delayMinutes > 3);
	const lowDelay = $derived(isRealtime && delayMinutes <= 3);
	const early = $derived(isRealtime && delayMinutes <= -1);
	const notOnTime = $derived(arriveBy ? highDelay : highDelay || early);
	const roughlyOnTime = $derived(arriveBy ? lowDelay : lowDelay && !early);
	const isValidDate = $derived(scheduled instanceof Date && !isNaN(scheduled.getTime()));

	const timeZoneOffset = $derived(
		isValidDate
			? new Intl.DateTimeFormat(language, { timeZone, timeZoneName: 'shortOffset' })
					.formatToParts(scheduled)
					.find((part) => part.type === 'timeZoneName')!.value
			: undefined
	);
	const isSameAsBrowserTimezone = $derived(() => {
		if (!isValidDate) return false;
		return (
			new Intl.DateTimeFormat(language, { timeZoneName: 'shortOffset' })
				.formatToParts(scheduled)
				.find((part) => part.type === 'timeZoneName')!.value == timeZoneOffset
		);
	});
</script>

<div class={cn('text-nowrap flex flex-col', className)} title={timeZoneOffset}>
	{#if variant == 'schedule'}
		<div>
			{formatTime(scheduled, timeZone)}
		</div>
		<div class="text-xs font-normal">{isSameAsBrowserTimezone() ? '' : timeZoneOffset}</div>
	{:else if variant === 'realtime-show-always' || (variant === 'realtime' && isRealtime)}
		<span class:text-destructive={notOnTime} class:text-green-600={roughlyOnTime}>
			{formatTime(t, timeZone)}
		</span>
		{#if variant === 'realtime-show-always' && !isSameAsBrowserTimezone()}
			<div class="text-xs font-normal">
				{isSameAsBrowserTimezone() ? '' : timeZoneOffset}
			</div>
		{/if}
	{/if}
</div>
