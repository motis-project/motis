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
		timeZone,
		isArrival
	}: {
		class?: string;
		timestamp: string;
		scheduledTimestamp: string;
		isRealtime: boolean;
		variant: 'schedule' | 'realtime' | 'realtime-show-always';
		queriedTime?: string | undefined;
		timeZone: string | undefined;
		isArrival?: boolean | undefined
	} = $props();

	const t = $derived(new Date(timestamp));
	const scheduled = $derived(new Date(scheduledTimestamp));
	const delayMinutes = $derived((t.getTime() - scheduled.getTime()) / 60000);
	const highDelay = $derived(isRealtime && delayMinutes > 3);
	const lowDelay = $derived(isRealtime && delayMinutes <= 3);
	const Early = $derived(isRealtime && delayMinutes < -1);
	const timeZoneOffset = $derived(
		new Intl.DateTimeFormat(language, { timeZone, timeZoneName: 'shortOffset' })
			.formatToParts(scheduled)
			.find((part) => part.type === 'timeZoneName')!.value
	);
	const isSameAsBrowserTimezone = $derived(
		new Intl.DateTimeFormat(language, { timeZoneName: 'shortOffset' })
			.formatToParts(scheduled)
			.find((part) => part.type === 'timeZoneName')!.value == timeZoneOffset
	);

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

<div class={cn('text-nowrap grid-cols-1 grid-rows-2', className)} title={timeZoneOffset}>
	{#if variant == 'schedule'}
		<div>
			{formatTime(scheduled, timeZone)}
			{weekday(scheduled)}
		</div>
		<div class="text-xs font-normal h-4">{isSameAsBrowserTimezone ? '' : timeZoneOffset}</div>
	{:else if variant === 'realtime-show-always' || (variant === 'realtime' && isRealtime)}
		<span class:text-destructive={isArrival? highDelay : Early } class:text-green-600={isArrival? lowDelay : (lowDelay && !Early)} class="bg-white">
			{formatTime(t, timeZone)}
			{weekday(t)}
		</span>
		{#if variant !== 'realtime-show-always'}
			<div class="text-xs font-normal h-4"></div>
		{/if}
	{/if}
</div>
