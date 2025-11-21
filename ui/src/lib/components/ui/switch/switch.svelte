<script lang="ts">
	import { Label, Switch } from 'bits-ui';
	import { cn } from '$lib/utils.js';

	let {
		checked = $bindable(false),
		disabled,
		label,
		id,
		class: className,
		onCheckedChange: onCheckedChange
	}: {
		checked: boolean;
		disabled?: boolean;
		label: string;
		id: string;
		class?: string;
		onCheckedChange?: (checked: boolean) => void;
	} = $props();
</script>

<div class="flex items-center space-x-3 my-2">
	<Switch.Root
		{id}
		{disabled}
		bind:checked
		class={cn(
			'peer inline-flex h-[21px] min-h-[21px] w-[45px] shrink-0 cursor-pointer items-center rounded-full px-[3px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-foreground focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-foreground data-[state=unchecked]:bg-border data-[state=unchecked]:shadow-mini-inset dark:data-[state=checked]:bg-foreground',
			className
		)}
		onCheckedChange={onCheckedChange}
	>
		<Switch.Thumb
			class="pointer-events-none block size-[15px] shrink-0 rounded-full bg-background transition-transform data-[state=checked]:translate-x-6 data-[state=unchecked]:translate-x-0 data-[state=unchecked]:shadow-mini dark:border dark:border-background/30 dark:bg-foreground dark:shadow-popover dark:data-[state=unchecked]:border"
		/>
	</Switch.Root>
	<Label.Root for={id} class="text-sm font-medium">{label}</Label.Root>
</div>
