<script lang="ts">
	import TriangleAlert from 'lucide-svelte/icons/triangle-alert';
	import type { Alert } from './api/openapi';
	import { t } from '$lib/i18n/translation';
	import ChevronDown from 'lucide-svelte/icons/chevron-down';
	import Time from './Time.svelte';

	let { alerts, timeZone }: { alerts: Alert[]; timeZone: string } = $props();

	let isOpen = $state(false);
</script>

<div class="flex flex-col gap-2 max-w-xs items-start">
	<button
		onclick={() => (isOpen = !isOpen)}
		class="p-1 bg-yellow-500 origin-top-left text-xs text-white font-bold rounded-lg hover:bg-yellow-600 transition-colors shadow-lg flex items-center"
	>
		<TriangleAlert class="size-4 mr-1" />
		{t.alertsAvailable}
		<ChevronDown
			size={20}
			class={`transition-transform duration-500 ${isOpen ? 'rotate-180' : ''}`}
		/>
	</button>
	{#each alerts as alert, i (i)}
		{#if isOpen}
			<div
				class="w-full items-center bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-2 rounded shadow"
			>
				<p class="font-bold overflow-hidden">{alert.headerText}</p>
				{#if alert.impactPeriod && alert.impactPeriod[0]?.start}
					<span class="text-sm font-bold">
						{t.from}
						<Time
							variant="schedule"
							class="text-sm font-bold inline-flex justify-center gap-2 align-middle items-center"
							{timeZone}
							isRealtime={false}
							timestamp={alert.impactPeriod[0]?.start ?? ''}
							scheduledTimestamp={alert.impactPeriod[0]?.start ?? ''}
						/>
					</span>
				{/if}
				{#if alert.impactPeriod && alert.impactPeriod.length > 1 && alert.impactPeriod[alert.impactPeriod.length - 1]?.end}
					<span class="text-sm block font-bold">
						{t.to}
						<Time
							variant="schedule"
							class="text-sm font-bold inline-flex justify-center gap-2 align-middle items-center"
							{timeZone}
							isRealtime={false}
							timestamp={alert.impactPeriod[alert.impactPeriod.length - 1].end ?? ''}
							scheduledTimestamp={alert.impactPeriod[0]?.start ?? ''}
						/>
					</span>
				{/if}
				<p class="text-sm overflow-hidden">{alert.descriptionText}</p>
			</div>
		{/if}
	{/each}
</div>
