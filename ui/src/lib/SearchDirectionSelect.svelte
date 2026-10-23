<script lang="ts">
	import * as Select from '$lib/components/ui/select';
	import { t } from '$lib/i18n/translation';

	let {
		arriveBy = $bindable(),
		leaveNow = $bindable(),
		onDirectionChange
	}: {
		arriveBy: boolean;
		leaveNow: boolean;
		onDirectionChange?: (dir: 'departure' | 'arrival') => void;
	} = $props();

	const label = $derived(leaveNow ? t.leaveNow : arriveBy ? t.arrival : t.departure);
</script>

<Select.Root
	type="single"
	bind:value={
		() => (leaveNow ? 'now' : arriveBy ? 'arrival' : 'departure'),
		(v) => {
			leaveNow = v === 'now';
			arriveBy = v === 'arrival';
		}
	}
	onValueChange={(v) => onDirectionChange?.(v === 'arrival' ? 'arrival' : 'departure')}
>
	<Select.Trigger class="w-auto gap-2">
		{label}
	</Select.Trigger>
	<Select.Content align="start">
		<Select.Item value="now" label={t.leaveNow}>{t.leaveNow}</Select.Item>
		<Select.Item value="departure" label={t.departure}>{t.departure}</Select.Item>
		<Select.Item value="arrival" label={t.arrival}>{t.arrival}</Select.Item>
	</Select.Content>
</Select.Root>
