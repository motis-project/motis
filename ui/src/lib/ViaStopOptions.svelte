<script lang="ts">
	import { X } from '@lucide/svelte';
	import AddressTypeahead from '$lib/AddressTypeahead.svelte';
	import { Button } from './components/ui/button';
	import NumberSelect from './NumberSelect.svelte';
	import { t } from './i18n/translation';
	import type { Location } from '$lib/Location';
	import { generateTimes } from './generateTimes';
	import { formatDurationMin } from './formatDuration';

	let {
		via = $bindable(),
		viaMinimumStay = $bindable(),
		viaLabels = $bindable()
	}: {
		via: undefined | Location[];
		viaMinimumStay: undefined | number[];
		viaLabels: Record<string, string>;
	} = $props();

	const possibleViaStayDurations = $derived([0, ...generateTimes(2 * 60 * 60)]);
	const viaMinimumStayOptions = $derived(
		possibleViaStayDurations.map((duration) => ({
			value: (duration / 60).toString(),
			label: formatDurationMin(duration / 60)
		}))
	);

	type Via = {
		match: Location;
		stay: number;
	};
	let vias = $state<Via[]>(
		via?.map(
			(_, i): Via => ({
				match: via![i],
				stay: viaMinimumStay?.[i] ?? 0
			})
		) ?? []
	);
	const add = () => {
		vias.push({ stay: 0, match: { label: '', match: undefined } });
	};
	const remove = (index: number) => {
		vias = vias.filter((_, i) => i !== index);
	};

	$effect(() => {
		const filtered = vias.filter((v) => v.match?.match?.id).map((v) => $state.snapshot(v));

		const oldVia = $state.snapshot(via);
		const nextVia = filtered.length > 0 ? filtered.map((f) => f.match) : undefined;
		if (JSON.stringify(nextVia) != JSON.stringify(oldVia)) {
			via = nextVia;
		}

		const oldViaMinimumStay = $state.snapshot(viaMinimumStay);
		const nextViaMinimumStay = filtered.length > 0 ? filtered.map((f) => f.stay) : undefined;
		if (JSON.stringify(nextViaMinimumStay) != JSON.stringify(oldViaMinimumStay)) {
			viaMinimumStay = nextViaMinimumStay;
		}

		const nextViaLabels: Record<string, string> = {};
		filtered.forEach((f, i) => {
			nextViaLabels[`viaLabel${i}`] = f.match.label;
		});
		Object.keys(viaLabels).forEach((key) => {
			if (!(key in nextViaLabels)) {
				delete viaLabels[key];
			}
		});
		Object.keys(nextViaLabels).forEach((key) => {
			if (viaLabels[key] !== nextViaLabels[key]) {
				viaLabels[key] = nextViaLabels[key];
			}
		});
	});
</script>

<div class="space-y-2">
	<div class="flex items-center justify-between">
		<div class="text-sm">
			{t.viaStops}
		</div>
		<Button variant="outline" onclick={add} disabled={vias.length >= 2}>
			{t.addViaStop}
		</Button>
	</div>
	<div class="space-y-2">
		{#each vias as _viaStop, index (index)}
			<div class="flex gap-2 items-start">
				<div class="grow flex flex-col gap-1">
					<div class="flex gap-2">
						<div class="grow">
							<AddressTypeahead
								placeholder={t.viaStop}
								name={`via-${index}`}
								bind:selected={vias[index].match}
								type="STOP"
							/>
						</div>
						<div class="w-24">
							<NumberSelect
								bind:value={vias[index].stay}
								possibleValues={viaMinimumStayOptions}
								labelFormatter={formatDurationMin}
							/>
						</div>
					</div>
				</div>
				<Button
					variant="ghost"
					size="icon"
					onclick={() => remove(index)}
					aria-label={t.removeViaStop}
				>
					<X class="size-4" />
				</Button>
			</div>
		{/each}
	</div>
</div>
