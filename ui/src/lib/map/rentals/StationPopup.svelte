<script lang="ts">
	import type { RentalProvider, RentalStation } from '$lib/api/openapi';
	import { Button } from '$lib/components/ui/button';
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
	{#if showActions && station.rentalUriWeb}
		<Button class="font-bold" variant="outline" href={station.rentalUriWeb} target="_blank">
			{t.rent}
		</Button>
	{/if}
</div>
