<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { formatLongDate, isSameDay } from './toDateTime';
	import { cn } from './utils';

	let {
		class: className,
		previous,
		timestamp,
		timeZone
	}: {
		class?: string;
		previous: Date | string | undefined;
		timestamp: string;
		timeZone: string | undefined;
	} = $props();
</script>

{#if previous && !isSameDay(new Date(previous), new Date(timestamp), timeZone)}
	<div class={cn('text-sm font-semibold text-muted-foreground', className)}>
		{t.onDate(formatLongDate(new Date(timestamp), timeZone))}
	</div>
{/if}
