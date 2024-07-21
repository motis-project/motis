<script lang="ts">
	import { Combobox } from 'bits-ui';
	import { Check } from 'lucide-svelte';

	let { placeholder }: { placeholder: string | undefined } = $props();

	let inputValue = $state('');
	let touchedInput = $state(false);

	const fruits = [
		{ value: 'mango', label: 'Mango' },
		{ value: 'watermelon', label: 'Watermelon' },
		{ value: 'apple', label: 'Apple' },
		{ value: 'pineapple', label: 'Pineapple' },
		{ value: 'orange', label: 'Orange' }
	];
	let filteredFruits = $derived(
		inputValue && touchedInput
			? fruits.filter((fruit) => fruit.value.includes(inputValue.toLowerCase()))
			: fruits
	);
	$inspect(filteredFruits);
</script>

<div>
	<Combobox.Root items={filteredFruits} bind:inputValue bind:touchedInput>
		<div class="relative">
			<Combobox.Input
				class="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
				{placeholder}
				aria-label={placeholder}
			/>
		</div>
		<Combobox.Content
			sideOffset={8}
			class="relative z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md outline-none"
		>
			{#each filteredFruits as fruit (fruit.value)}
				<Combobox.Item
					class="relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-4 pr-2 text-sm outline-none data-[disabled]:pointer-events-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground data-[disabled]:opacity-50"
					value={fruit.value}
					label={fruit.label}
				>
					{fruit.label}
					<Combobox.ItemIndicator class="ml-auto" asChild={false}>
						<Check />
					</Combobox.ItemIndicator>
				</Combobox.Item>
			{:else}
				<span class="block px-5 py-2 text-sm text-muted-foreground"> No results found </span>
			{/each}
		</Combobox.Content>
		<Combobox.HiddenInput name="favoriteFruit" />
	</Combobox.Root>
</div>
