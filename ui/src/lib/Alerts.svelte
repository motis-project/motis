<script lang="ts">
	import Info from 'lucide-svelte/icons/info';
	import { Button } from './components/ui/button';
	import type { Alert } from '$lib/api/openapi';

	const {
		alerts = [],
		variant = 'icon'
	}: {
		alerts?: Alert[];
		variant?: 'icon' | 'full';
	} = $props();
</script>

{#if alerts.length > 0}
	{#if variant === 'full'}
		<div class="w-full pr-2 md:pr-4 mt-2">
			<Button
				class="w-full justify-start flex flex-col items-start bg-blue-100 dark:bg-blue-950 shadow-none"
				variant="outline"
			>
				<div class="font-bold flex gap-2 items-center text-blue-700 dark:text-blue-500">
					<Info /> Informationen
					{#if alerts.length > 1}
						<span class="text-muted-foreground font-normal text-sm">
							+{alerts.length - 1} mehr
						</span>
					{/if}
				</div>
				<span class="font-normal text-muted-foreground overflow-hidden text-ellipsis w-full">
					{alerts[0].descriptionText}
				</span>
			</Button>
		</div>
	{:else}
		<Button class="ml-2 rounded-full" variant="outline" size="sm">
			<Info />
		</Button>
	{/if}
{/if}
