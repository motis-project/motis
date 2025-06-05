<script lang="ts" module>
	import { tv } from 'tailwind-variants';

	export const tabVariants = tv({
		base: 'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 disabled:pointer-events-none disabled:opacity-50',
		variants: {
			variant: {
				default: 'bg-transparent',
				outline:
					'border-input hover:bg-accent hover:text-accent-foreground border bg-transparent shadow-sm'
			},
			size: {
				default: 'h-9 min-w-9 px-3',
				sm: 'h-8 min-w-8 px-2',
				lg: 'h-10 min-w-10 px-3'
			}
		},
		defaultVariants: {
			variant: 'default',
			size: 'default'
		}
	});
</script>

<script lang="ts">
</script>
