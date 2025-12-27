<script lang="ts">
	import { t } from '$lib/i18n/translation';
	import { LoaderCircle } from '@lucide/svelte';
	import ErrorMessage from '$lib/ErrorMessage.svelte';
	import type { IsochronesOptions } from '$lib/map/IsochronesShared';

	let {
		options
	}: {
		options: IsochronesOptions;
	} = $props();
</script>

<div>
	{#if options.status == 'WORKING'}
		<div class="flex items-center justify-center w-full">
			<LoaderCircle class="animate-spin w-12 h-12 m-4" />
		</div>
	{/if}
	{#if options.status == 'EMPTY'}
		<ErrorMessage message={t.isochrones.noData} status={404} />
	{/if}
	{#if options.status == 'FAILED'}
		<ErrorMessage message={options.errorMessage!} status={options.errorCode} />
	{/if}
</div>
