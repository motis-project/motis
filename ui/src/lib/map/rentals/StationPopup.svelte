<script lang="ts">
	import type { RentalProvider, RentalStation } from '$lib/api/openapi';
	import { t } from '$lib/i18n/translation';

	let {
		provider,
		station,
		showActions = false
	}: {
		provider: RentalProvider;
		station: RentalStation;
		showActions?: boolean;
	} = $props();

	let rentUrl = $derived(station.rentalUriWeb ?? provider.purchaseUrl ?? provider.url ?? null);
</script>

<div class="space-y-3 text-sm leading-tight text-foreground">
	<div class="space-y-1">
		<div class="font-semibold">{station.name}</div>
		<div>
			{t.sharingProvider}: {#if provider.url}
				<a
					href={provider.url}
					target="_blank"
					class="text-blue-600 dark:text-blue-300 hover:underline"
				>
					{provider.name}
				</a>
			{:else}
				{provider.name}
			{/if}
		</div>
	</div>
	{#if showActions && rentUrl}
		<a
			href={rentUrl}
			target="_blank"
			class="inline-flex items-center justify-center rounded-md bg-blue-600 px-3 py-1.5 text-sm font-semibold text-white no-underline hover:bg-blue-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500"
		>
			{t.rent}
		</a>
	{/if}
</div>
