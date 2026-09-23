<script lang="ts">
    import { type ElevationProfile } from "@motis-project/motis-client";

    let { profile }: { profile: ElevationProfile } = $props();

    interface IResult {
        distance: number;
        elevation: number;
    }

    const sqrdDistance = (a: number[], b: number[]): number => {
        return Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2);
    }

    let maxDistance = 0;
    const processData = (points: number[]): IResult[] => {
        maxDistance = 0;
        let reduced = points.reduce((acc: number[][], v, i) => {
            acc[acc.length - 1].push(v);
            if ((i+1) % 3 == 0 && i < points.length - 1) {
                acc.push([]);
            }

            return acc;
        }, [[]]);
        
        return reduced.map((v: number[], i: number): IResult => {
            if (i == 0) {
                return { distance: 0, elevation: v[2]};
            }

            maxDistance += sqrdDistance(v, reduced[i-1]);
            return { distance: maxDistance, elevation: v[2]};
        });
    };

    let width = 100;
    let height = 100;
    const xScale = (d: number): number => {
        return d / maxDistance * width;
    }
    const yScale = (d: number): number => {
        return (d - profile.min) / profile.max * height;
    }

    console.log(`maxDistance: ${maxDistance}`)
    console.log(`minEle: ${profile.min} | maxEle: ${profile.max}`)

    let linePath = processData(profile.points).reduce((acc: string, v: IResult, i: number): string => {
        const x = xScale(v.distance);
        const y = yScale(v.elevation);

        console.log(`x: ${x} | y: ${y}`)

        return (i == 0) ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
    }, '');
</script>

<div class="chart-containers">
    <svg viewBox="0 0 {width} {height}">
        <path d={linePath} fill="none" stroke="#30919c" stroke-width="2" />
        <path d="M 0 {height} L {width} {height}" fill="none" stroke="#000000" stroke-width="4" />
        <path d="M 0 0 L 0 {height}" fill="none" stroke="#000000" stroke-width="4" />
        <text x={width} y={height - 10} font-size=10 fill="#64748b" text-anchor="end">{maxDistance.toFixed(1)} m</text>
    </svg>
</div>