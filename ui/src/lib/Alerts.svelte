<script lang="ts">
	import { Info, ChevronRight } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import { buttonVariants } from './components/ui/button';
	import type { Alert } from '@motis-project/motis-client';
	import { formatDateTime, getTz } from './toDateTime';
	import { cn } from './utils';
	import { t } from './i18n/translation';

	const {
		alerts = [],
		variant = 'icon',
		tz
	}: {
		alerts?: Alert[];
		variant?: 'icon' | 'full';
		tz: string | undefined;
	} = $props();
</script>

{#if alerts.length > 0}
	<Dialog.Root>
		<Dialog.Trigger class="max-w-full pr-4  {variant == 'full' ? 'pt-2' : 'ml-2'}">
			{#if variant === 'full'}
				<div
					class={cn(
						buttonVariants({ variant: 'outline' }),
						'max-w-full flex items-center bg-blue-50 dark:bg-blue-950 shadow-none'
					)}
				>
					<div class="flex flex-col gap-1 overflow-hidden">
						<div class="font-bold flex gap-2 items-center text-blue-700 dark:text-blue-500">
							<Info />
							{t.alerts.information}
							{#if alerts.length > 1}
								<span class="text-muted-foreground font-normal">
									+{alerts.length - 1}
									{t.alerts.more}
								</span>
							{/if}
						</div>
						<span class="font-normal text-muted-foreground overflow-hidden text-ellipsis w-full">
							{alerts[0].descriptionText || alerts[0].headerText}
						</span>
					</div>
					<ChevronRight class="size-4" />
				</div>
			{:else}
				<Info />
			{/if}
		</Dialog.Trigger>
		<Dialog.Content>
			<Dialog.Header>
				<Dialog.Description class="space-y-4">
					{#each alerts as alert, i (i)}
						<div class="last:mb-0 text-justify">
							<h3 class="font-bold text-blue-700 dark:text-blue-500 mb-1 flex items-center gap-2">
								<Info class="size-5" />{alert.headerText}
							</h3>
							{#if alert.impactPeriod}
								{#each alert.impactPeriod as impactPeriod, j (j)}
									{@const start = new Date(impactPeriod.start ?? 0)}
									{@const end = new Date(impactPeriod.end ?? 0)}
									<p>
										<strong>{t.alerts.validFrom}:</strong>
										{formatDateTime(start, tz)}
										<strong>{t.alerts.until}</strong>
										{formatDateTime(end, tz)}
										<span class="text-xs font-normal">{getTz(start, tz)}</span>
									</p>
								{/each}
							{/if}
							{#if alert.causeDetail}
								<p>{alert.causeDetail}</p>
							{/if}
							{#if alert.descriptionText}
								<p>{alert.descriptionText}</p>
							{/if}
						</div>
					{/each}
				</Dialog.Description>
			</Dialog.Header>
		</Dialog.Content>
	</Dialog.Root>
{/if}
