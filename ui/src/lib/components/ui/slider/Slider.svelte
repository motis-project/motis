<script lang="ts">
  import { Slider } from "bits-ui";
  import cn from "clsx";
    
  let {
    value = $bindable(),
    disabled,
    step,
    min,
    max
  } : {
    value: number,
    disabled?:boolean,
    step?:number,
    min?:number,
    max?:number
  } = $props();
</script>
 
<div class="w-full md:max-w-[180px]">
  <Slider.Root
    type="single"
    {min}
    {max}
    {step}
    {disabled}
    bind:value
    class="relative flex w-full touch-none select-none items-center"
  >
    <span
      class="bg-dark-10 relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full"
    >
      <Slider.Range class="bg-foreground absolute h-full" />
    </span>
    <Slider.Thumb
      index={0}
      class={cn(
        "border-border-input bg-background hover:border-dark-40 focus-visible:ring-foreground dark:bg-foreground dark:shadow-card data-active:border-dark-40 focus-visible:outline-hidden data-active:scale-[0.98] block size-[15px] cursor-pointer rounded-full border shadow-sm transition-colors focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
      )}
    />
  </Slider.Root>
</div>