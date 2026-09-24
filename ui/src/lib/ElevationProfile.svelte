<script lang="ts">
    import { type ElevationProfile } from "@motis-project/motis-client";
	import { points } from "@turf/helpers";

    let { profile }: {profile: ElevationProfile} = $props();

    interface IResult {
        distance: number;
        elevation: number;
    }

    const latFactor = 111.13;  // distance (km) for 1 degree latitude
    const distance = (lat1: number, lng1: number, lat2: number, lng2: number): number => {
        const lngFactor = Math.cos(lat1 * Math.PI / 180) * latFactor;
        const sqrdDist = Math.pow((lat1 - lat2) * latFactor, 2) + Math.pow((lng1 - lng2) * lngFactor, 2);
        return Math.sqrt(sqrdDist);
    }

    let maxDistance = 0;
    let minResolution: number;
    let maxResolution = 0;
    const processData = (points: number[]): IResult[] => {
        let res: IResult[] = [{distance: 0, elevation: points[2]}];
        for (let i = 3; i < points.length - 2; i += 3) {
            const d = distance(points[i], points[i+1], points[i-3], points[i-2])
            minResolution = (minResolution) ? Math.min(minResolution, d) : d;
            maxResolution = Math.max(maxResolution, d);
            maxDistance += d;
            res.push({distance: maxDistance, elevation: points[i+2]});
        }

        return res;
    };

    const width = 400;
    const height = 400;
    const fontSize = 20;
    const xPadding = Math.max(40, `${profile.max}m`.length * fontSize * 0.6 + 5);
    const yPadding = 40;
    const xScale = (d: number): number => {
        return d / maxDistance * (width - xPadding * 2) + xPadding;
    }
    const yScale = (d: number): number => {
        return height - yPadding - (d - profile.min) / (profile.max - profile.min) * (height - yPadding * 2);
    }

    let linePath = processData(profile.points).reduce((acc: string, v: IResult, i: number): string => {
        const x = xScale(v.distance);
        const y = yScale(v.elevation);

        return (i == 0) ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
    }, '');

    console.log(`minRes=${minResolution * 1000} | maxRes=${maxResolution * 1000} | avgRes=${maxDistance / (profile.size || profile.points.length / 3) * 1000} | targetRes=${profile.resolution}`);
</script>

<div class="flex flex-col items-center">
    <svg viewBox="0 0 {width} {height}" class="w-full h-auto">
        <!-- x-Axis -->
        <path class="axis-line" d="M {xPadding} {height - yPadding} L {width - xPadding} {height - yPadding}" />
        <text class="label" x={xPadding} y={height - yPadding + fontSize} font-size={fontSize} fill="#64748b" text-anchor="middle">0km</text>
        <text class="label" x={width / 2} y={height - yPadding + fontSize} font-size={fontSize} fill="#64748b" text-anchor="middle">{(maxDistance / 2).toFixed(2)}km</text>
        <text class="label" x={width - xPadding} y={height - yPadding + fontSize} font-size={fontSize} fill="#64748b" text-anchor="middle">{maxDistance.toFixed(2)}km</text>
        <!-- Median -->
        {#if profile.median}
            {@const y = yScale(profile.median)}    
            <path class="median-line" d="M {xPadding} {y} L {width - xPadding} {y}" />
            <text class="label" x={xPadding - 5} y={y} font-size={fontSize} fill="#64748b" text-anchor="end">{profile.median}m</text>
        {/if}
        <!-- y-Axis -->
        <path class="axis-line" d="M {xPadding + 0.5} {yPadding} L {xPadding + 0.5} {height - yPadding}" />
        <text class="label" x={xPadding - 5} y={height - yPadding} font-size={fontSize} fill="#64748b" text-anchor="end">{profile.min}m</text>
        <text class="label" x={xPadding - 5} y={yPadding} font-size={fontSize} fill="#64748b" text-anchor="end">{profile.max}m</text>
        <!-- Profile -->
        <path d={linePath} fill="none" stroke="#30919c" stroke-width="1" />
    </svg>
    <div class="mt-2 text-center text-sm text-slate-500">
        median={profile.median}m
        N={profile.size}
    </div>
</div>

<style>
    .label {
        fill: "#64748b";
    }

    .axis-line {
        fill: none;
        stroke: #000000;
        stroke-width: 1px;
    }

    .median-line {
        fill: none;
        stroke: #000000;
        stroke-width: 1px;
        opacity: 0.4
    }
</style>