<script lang="ts">
	import maplibregl, { GeoJSONSource, Map, type LngLatBoundsLike, type TextureFilter } from "maplibre-gl";
    import { circle } from "@turf/circle";
    import { featureCollection } from "@turf/helpers";
    import { union } from "@turf/union";
	import combine from "@turf/combine";

    interface Pos {
        lat: number;
        lng: number;
        duration: number;
        name?: string;
    };

	let {
        map,
		bounds = $bindable(),
		isochronesData = $bindable(),
	} : {
        map: Map | undefined;
		bounds: LngLatBoundsLike | undefined;
		isochronesData: Pos[];
    } = $props();

    const name = "Isochrones";
	const factor = 0.06;  // TODO Calculate factor for duration ~~> radius (in kilometers)

    let loaded = false;

	const box = $derived(maplibregl.LngLatBounds.convert(bounds ? bounds : [[0, 0], [0, 0]]));

    function reachable_kilemeters(pos: Pos) {
        return pos.duration * factor;
    }
    function is_visible(data: Pos) {
        const r = reachable_kilemeters(data);
        return box._sw.lat <= data.lat + r && data.lat - r <= box._ne.lat
			&& box._sw.lng <= data.lng + r && data.lng - r <= box._ne.lat;
    }

	const circles = $derived.by(() => {
		// const visible = isochronesData.filter(is_visible);
		const visible = isochronesData;
        const max_steps = Math.floor(10_000 / (visible.length + 1));
        const steps = Math.max(Math.min(max_steps, 64), 3);
        console.log(`STEPS: ${max_steps}  //  ${steps}`);
        return visible
            .map((data) => {
                const r = reachable_kilemeters(data);
                return circle(
                    [data.lng, data.lat],
                    r,
                    {
                        steps: steps,
                        units: "kilometers",
                        properties: {
                            "name": data.name ?? 'unknown',
                        }
                    }
                );
            })
    });

    $effect(() => {
        if (!map) {
            return;
        }
        if (!loaded) {
            map.addSource(name, {type: "geojson", data: ''});
            // map.addSource(name, {type: "geojson", data: {geometry: {type: "MultiPolygon", coordinates: []}, type: "Feature", properties: {}}});
            map.addLayer({
                id: name,
                type: "fill",
                source: name,
                paint: {
                    "fill-color": "magenta",
                    "fill-opacity": 0.25,
                }
            });
            loaded = true;
        }
        const coll = combine(featureCollection([...circles]));
        // const coll = union(featureCollection([...circles]));
        if (coll) {
            // (map.getSource(name) as GeoJSONSource).setData(coll.geometry);
            (map.getSource(name) as GeoJSONSource).setData(coll);
        }
    })
</script>
