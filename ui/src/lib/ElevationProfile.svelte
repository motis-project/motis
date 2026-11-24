<script lang="ts">
	import { type ElevationProfile } from '@motis-project/motis-client';

	let { profile }: { profile: ElevationProfile } = $props();
    let rootElement: HTMLElement | undefined = $state();
    let fontSize: number = $state(12);
    let width: number = $state(0);
    let height: number = $derived(width / 2);

    $effect(() => {
        if (!rootElement) return;

        const updateSize = () => {
            if (!rootElement) return;

            const computed = window.getComputedStyle(rootElement);
            const parsed = parseFloat(computed.fontSize);
            if (!isNaN(parsed) && parsed > 0) {
                fontSize = parsed;
            }

            const newWidth = rootElement.getBoundingClientRect().width;
            if (newWidth > 0) {
                width = newWidth;
            }
        };
        updateSize();
        const observer = new ResizeObserver(() => requestAnimationFrame(updateSize));
        observer.observe(rootElement);

        const details = rootElement.closest("details");
        details?.addEventListener("toggle", updateSize);

        return () => {
            observer.disconnect(); 
            details?.removeEventListener("toggle", updateSize)
        };
    });

	interface IResult {
		distance: number;
		elevation: number;
	}

	const latFactor = 111.13; // distance (km) for 1 degree latitude
	const distance = (lat1: number, lng1: number, lat2: number, lng2: number): number => {
		const lngFactor = Math.cos((lat1 * Math.PI) / 180) * latFactor;
		const sqrdDist =
			Math.pow((lat1 - lat2) * latFactor, 2) + Math.pow((lng1 - lng2) * lngFactor, 2);
		return Math.sqrt(sqrdDist);
	};

	const processData = (points: number[]): IResult[] => {
		let res: IResult[] = [{ distance: 0, elevation: points[2] }];
        let dist_acc = 0;
		for (let i = 3; i < points.length - 2; i += 3) {
			const d = distance(points[i], points[i + 1], points[i - 3], points[i - 2]);
			dist_acc += d;
            res.push({ distance: dist_acc, elevation: points[i + 2] });
		}

		return res;
	};

    const xPadding = $derived(`${profile.max}`.length * fontSize * 1.2 + 5);
	const yPadding = $derived(fontSize * 1.4 + 5);
	const xScale = (d: number): number => {
		return d * (width - xPadding * 2) + xPadding;
	};
	const yScale = (d: number): number => {
		return height - yPadding - d * (height - yPadding * 2);
	};

	let chart = $derived.by(() => {
        let processed = processData(profile.points);
        let dist = processed[processed.length - 1].distance;
        let path = processed.reduce(
            (acc: string, v: IResult, i: number): string => {
                const x = xScale(v.distance / dist);
                const y = yScale(v.elevation / profile.max);

                return i == 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
            },
            ''
	    );
        return {path: path, dist: dist};
    });
</script>

<div bind:this={rootElement} class="block w-full">
    {#if width > 0 && height > 0}
    {@const maxDistance = chart.dist}
    {@const linePath = chart.path}
	<svg {width} {height} viewBox="0 0 {width} {height}" class="block w-full aspect-[2/1] overflow-visible">
		<!-- x-Axis -->
		<path
			class="axis-line"
			d="M {xPadding} {height - yPadding} L {width - xPadding} {height - yPadding}"
		/>
		<text
            class="label"
			x={xPadding}
			y={height - yPadding + fontSize}
			text-anchor="middle">0km</text
		>
		<text
            class="label"
			x={width / 2}
			y={height - yPadding + fontSize}
			text-anchor="middle">{(maxDistance / 2).toFixed(2)}km</text
		>
		<text
            class="label"
			x={width - xPadding}
			y={height - yPadding + fontSize}
			text-anchor="middle">{maxDistance.toFixed(2)}km</text
		>
		<!-- Median -->
		<!-- {#if profile.median}
			{@const yMed = yScale(profile.median)}
			<path class="median-line" d="M {xPadding} {yMed} L {width - xPadding} {yMed}" />
			<text class="label" x={xPadding - 5} y={yMed} text-anchor="end"
				>{profile.median}m</text
			>
		{/if} -->
		<!-- y-Axis -->
		<path
			class="axis-line"
			d="M {xPadding + 0.5} {yPadding} L {xPadding + 0.5} {height - yPadding}"
		/>
		<text
			class="label"
			x={xPadding - 5}
			y={height - yPadding}
			text-anchor="end">0m</text
		>
        <!-- Minimum -->
        <path
            class="median-line"
            d="M {xPadding} {yScale(profile.min / profile.max)} L {width - xPadding} {yScale(profile.min / profile.max)}"
        />
        <text
            class="label"
            x={xPadding - 5} y={yScale(profile.min / profile.max)}
            text-anchor="end" dominant-baseline="hanging"
        >{profile.min}m</text>
        <!-- Maximum -->
        <path
            class="median-line" 
            d="M {xPadding} {yScale(1.0)} L {width - xPadding} {yScale(1.0)}"
        />
        <text
            class="label"
			x={xPadding - 5} y={yScale(1.0)}
			text-anchor="end" dominant-baseline="alphabetic">{profile.max}m</text
		>
		<!-- Profile -->
		<path d={linePath} fill="none" stroke="#30919c" stroke-width="1" />
	</svg>
	<!-- <div class="mt-2 text-center text-sm text-slate-500">
		median={profile.median}m N={profile.size}
		up={profile.up}m down={profile.down}m
	</div> -->
    {/if}
</div>

<style>
    .label {
        fill: currentColor;
        font-size: inherit;
        font-family: inherit;
    }

	.axis-line {
		fill: none;
		stroke: currentColor;
		stroke-width: 1px;
	}

	.median-line {
		fill: none;
		stroke: currentColor;
        stroke-dasharray: 2 4;
		stroke-width: 1px;
		opacity: 0.4;
	}
</style>
