<script lang="ts">
	import maplibregl, { GeoJSONSource, Map, type Feature, type LngLatBoundsLike, type TextureFilter } from "maplibre-gl";
    import { circle } from "@turf/circle";
    import { featureCollection, multiPolygon, polygon } from "@turf/helpers";
    import { union } from "@turf/union";
	import combine from "@turf/combine";
	import intersect from "@turf/intersect";
	import difference from "@turf/difference";
	import { assert } from "vitest";
	import flatten from "@turf/flatten";
	import type { PolygonLayer } from "@deck.gl/layers";

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

    type MyType = ReturnType<typeof union>;
    function my_union(d: MyType[]): MyType {
        if (d.length == 0) {
            return null;
        }
        if (d.length == 1) {
            return d[0];
        }
        const l = Math.floor(d.length / 2);
        const c1 = d.slice(0, l);
        const c2 = d.slice(l);
        const u1 = my_union(c1);
        const u2 = my_union(c2);
        if (u1 === null) {
            return u2;
        }
        if (u2 === null) {
            return u1;
        }
        const u = union(featureCollection([u1, u2]));
        return u;
    }

    function fasterUnion(allGeometries: NonNullable<MyType>[]) {
  const mid = Math.floor(allGeometries.length / 2);
  let group1 = allGeometries.slice(0, mid);
  let group2 = allGeometries.slice(mid);

  while (group1.length > 1) {
    group1 = unionGroup(group1);
  }
  while (group2.length > 1) {
    group2 = unionGroup(group2);
  }

  let result;
  if (group1.length === 1 && group2.length === 1) {
    result = union(featureCollection([group1[0], group2[0]]));
  } else if (group1.length === 1) {
    result = group1[0];
  } else {
    result = group2[0];
  }

  return result;
}

function unionGroup(group: NonNullable<MyType>[]): NonNullable<MyType>[] {
  let newGroup = [];
  for (let i = 0; i < group.length; i += 2) {
    let a = group[i];
    let b = i + 1 < group.length ? group[i + 1] : null;
    if (b) {
    //   newGroup.push(union(a, b));
      const t = union(featureCollection([a, b]));
      if (t) {
        newGroup.push(t!);
      }
    } else {
      newGroup.push(a);
    }
  }
  return newGroup;
}

	const circles = $derived.by(() => {
		// const visible = isochronesData.filter(is_visible);
		const visible = isochronesData;
        // const max_steps = Math.floor(10_000 / (visible.length + 1));
        // const steps = Math.max(Math.min(max_steps, 64), 3);
        // console.log(`STEPS: ${max_steps}  //  ${steps}`);
        return visible
            .map((data) => {
                const r = reachable_kilemeters(data);
                return circle(
                    [data.lng, data.lat],
                    r,
                    {
                        // steps: steps,
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
        const coll = fasterUnion(circles);
        // const coll = my_union(circles);
        // const coll = combine(featureCollection([...circles]));
        // const coll = union(featureCollection([...circles]));
        if (coll) {
            // (map.getSource(name) as GeoJSONSource).setData(coll.geometry);
            (map.getSource(name) as GeoJSONSource).setData(coll);
        }
    })
</script>
