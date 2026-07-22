import type {
	HillshadeLayerSpecification,
	RasterDEMSourceSpecification,
	StyleSpecification
} from 'maplibre-gl';
import { LEVEL_MIN_ZOOM } from '../constants';
export const colors = {
	light: {
		background: '#f8f4f0',

		water: '#99ddff',
		waterText: 'hsl(205, 56%, 38%)',
		glacier: 'rgb(255, 255, 255)',
		damPier: 'rgb(249, 244, 238)',
		bridgeArea: 'rgb(244, 239, 233)',
		bridgeCasing: 'rgb(217, 217, 217)',
		bicycleRoad: 'rgb(239, 249, 255)',
		housenumber: 'hsla(33, 48%, 15%, 0.6)',
		rail: '#a8a8a8',
		railOutline: 'rgb(177, 187, 196)',
		railCore: 'rgb(197, 204, 211)',
		subwayOutline: 'rgb(166, 184, 199)',
		subwayCore: 'rgb(188, 202, 213)',
		pedestrian: '#e8e7eb',
		ferryRoute: 'rgba(102, 102, 255, 0.5)',

		sport: '#d0f4be',
		sportOutline: '#b3e998',

		building: '#ded7d3',
		buildingOutline: '#cfc8c4',

		landuseComplex: '#f0e6d1',
		landuseCommercial: 'hsla(0, 60%, 87%, 0.23)',
		landuseIndustrial: '#e0e2eb',
		landuseResidential: '#ece7e4',
		landuseRetail: 'hsla(0, 60%, 87%, 0.23)',
		landuseConstruction: '#a9a9a9',
		landuseAgriculture: 'rgb(240, 231, 209)',
		siteHospital: 'rgb(255, 102, 102)',

		landusePark: '#b8ebad',
		landuseNatureLight: '#cdedc0',
		landuseNatureHeavy: '#90ee90',
		landuseCemetery: '#e0e4dd',
		landuseBeach: '#fffcd3',

		indoorCorridor: '#fdfcfa',
		indoor: '#d4edff',
		indoorOutline: '#808080',
		indoorText: '#333333',

		publicTransport: 'rgba(218,140,140,0.3)',

		footway: 'rgb(252, 251, 250)',
		footwayOutline: 'rgb(206, 202, 199)',
		track: '#c0aa77',
		livingStreet: 'rgb(248, 245, 244)',
		footpath: 'rgb(160, 160, 160)',
		cycleway: 'rgb(100, 145, 205)',
		unclassified: 'rgb(232, 230, 227)',
		steps: '#ff4524',

		elevatorOutline: '#808080',
		elevator: '#bcf1ba',

		roadBackResidential: '#ffffff',
		roadBackNonResidential: '#ffffff',

		motorway: '#ffb366',
		motorwayLink: '#f7e06e',
		primarySecondary: '#fffbf8',
		linkTertiary: '#ffffff',
		residential: '#ffffff',
		road: '#ffffff',

		townText: '#333333',
		townTextHalo: 'rgba(255, 255, 255, 0.8)',
		text: '#333333',
		textHalo: 'rgba(255, 255, 255, 0.8)',
		citiesText: 'hsl(0, 0%, 20%)',
		citiesTextHalo: 'rgba(255, 255, 255, 0.8)',

		shield: 'shield'
	},
	dark: {
		background: '#292929',

		water: '#1f2830',
		waterText: 'hsl(205, 40%, 65%)',
		glacier: 'hsl(0, 0%, 10%)',
		damPier: 'hsl(33, 48%, 5%)',
		bridgeArea: 'rgb(22, 17, 11)',
		bridgeCasing: 'rgb(38, 38, 38)',
		bicycleRoad: 'hsl(203, 100%, 3%)',
		housenumber: 'hsla(33, 48%, 90%, 0.6)',
		rail: '#808080',
		railOutline: 'hsl(208, 14%, 27%)',
		railCore: 'rgb(44, 52, 59)',
		subwayOutline: 'hsl(207, 23%, 28%)',
		subwayCore: 'rgb(42, 55, 67)',
		pedestrian: '#292929',
		ferryRoute: 'rgba(58, 77, 139, 0.5)',

		sport: '#272525',
		sportOutline: '#272525',

		building: '#1F1F1F',
		buildingOutline: '#1A1A1A',

		landuseComplex: '#292929',
		landuseCommercial: '#292929',
		landuseIndustrial: '#353538',
		landuseResidential: '#292929',
		landuseRetail: '#292929',
		landuseConstruction: 'hsl(0, 0%, 34%)',
		landuseAgriculture: 'hsl(43, 51%, 12%)',
		siteHospital: 'hsl(0, 100%, 30%)',

		landusePark: '#18221f',
		landuseNatureLight: '#1e2322',
		landuseNatureHeavy: '#1a2020',
		landuseCemetery: '#202423',
		landuseBeach: '#4c4b3e',

		indoorCorridor: '#494949',
		indoor: '#1a1a1a',
		indoorOutline: '#0d0d0d',
		indoorText: '#eeeeee',

		publicTransport: 'rgba(89, 45, 45, 0.405)',

		footway: 'rgb(30, 30, 30)',
		footwayOutline: 'rgb(62, 62, 62)',
		track: '#483d24',
		livingStreet: 'rgb(46, 46, 46)',
		footpath: 'rgb(80, 80, 80)',
		cycleway: 'rgb(110, 150, 205)',
		unclassified: 'rgb(50, 50, 50)',
		steps: '#70504b',

		elevatorOutline: '#808080',
		elevator: '#3b423b',

		roadBackResidential: '#414141',
		roadBackNonResidential: '#414141',

		motorway: '#414141',
		motorwayLink: '#414141',
		primarySecondary: '#414141',
		linkTertiary: '#414141',
		residential: '#414141',
		road: '#414141',

		text: '#9a9a9a',
		textHalo: 'rgba(0, 0, 0, 0.8)',
		townText: '#bebebe',
		townTextHalo: 'rgba(0, 0, 0, 0.8)',
		citiesText: '#bebebe',
		citiesTextHalo: 'rgba(0, 0, 0, 0.8)',

		shield: 'shield-dark'
	}
};

function getUrlBase(url: string): string {
	const { origin, pathname } = new URL(url);
	return origin + pathname.slice(0, pathname.lastIndexOf('/') + 1);
}

// this doesn't escape {}-parameters
function getAbsoluteUrl(base: string, relative: string): string {
	return getUrlBase(base) + relative;
}

// Shortbread `streets` layer: `kind` is the base highway class, links carry
// `link=true` instead of a `*_link` kind, railways share the layer with
// `rail=true` and aeroways appear as kind runway/taxiway.
export const getStyle = (
	theme: 'light' | 'dark',
	level: number,
	staticBaseUrl: string,
	apiBaseUrl: string,
	withHillshades: boolean
): StyleSpecification => {
	const c = colors[theme];
	const hillshadeSources: StyleSpecification['sources'] = withHillshades
		? {
				hillshadeSource: {
					type: 'raster-dem',
					tiles: [getAbsoluteUrl(apiBaseUrl, 'mapterhorn/{z}/{x}/{y}.webp')],
					attribution: "<a href='https://mapterhorn.com/attribution'>© Mapterhorn</a>",
					bounds: [-180, -85.0511287, 180, 85.0511287],
					encoding: 'terrarium',
					tileSize: 512
				} satisfies RasterDEMSourceSpecification
			}
		: {};
	const hillshadeLayers: HillshadeLayerSpecification[] = withHillshades
		? [
				{
					id: 'hillshade',
					type: 'hillshade',
					source: 'hillshadeSource',
					paint: {
						'hillshade-exaggeration': 0.33
					}
				}
			]
		: [];
	return {
		version: 8,
		sources: {
			osm: {
				type: 'vector',
				tiles: [getAbsoluteUrl(apiBaseUrl, 'tiles/{z}/{x}/{y}.mvt')],
				maxzoom: 20,
				attribution: '© OpenStreetMap contributors'
			},
			...hillshadeSources
		},
		glyphs: getAbsoluteUrl(staticBaseUrl, 'glyphs/{fontstack}/{range}.pbf'),
		// MOTIS icons (elevator, shields) from the default sprite; the `basics`
		// sprite (from VersaTiles) carries fill patterns (basics:pattern-*).
		sprite: [
			{ id: 'default', url: getAbsoluteUrl(staticBaseUrl, 'sprite_sdf') },
			{ id: 'basics', url: getAbsoluteUrl(staticBaseUrl, 'sprites/basics/sprites') }
		],
		layers: [
			{
				id: 'background',
				type: 'background',
				paint: { 'background-color': c.background }
			},
			{
				id: 'coastline',
				type: 'fill',
				source: 'osm',
				'source-layer': 'coastline',
				paint: { 'fill-color': c.water }
			},
			{
				id: 'landuse_park',
				type: 'fill',
				source: 'osm',
				'source-layer': 'land',
				filter: [
					'in',
					'kind',
					'park',
					'garden',
					'playground',
					'miniature_golf',
					'golf_course',
					'village_green',
					'recreation_ground',
					'greenhouse_horticulture',
					'allotments'
				],
				paint: {
					'fill-color': c.landusePark,
					'fill-opacity': ['interpolate', ['linear'], ['zoom'], 8, 0.6, 10, 1]
				}
			},
			{
				id: 'landuse',
				type: 'fill',
				source: 'osm',
				'source-layer': 'land',
				filter: [
					'!in',
					'kind',
					'park',
					'garden',
					'playground',
					'miniature_golf',
					'golf_course',
					'village_green',
					'recreation_ground',
					'greenhouse_horticulture',
					'allotments',
					'public_transport'
				],
				paint: {
					'fill-color': [
						'match',
						['get', 'kind'],
						'commercial',
						c.landuseCommercial,
						['industrial', 'railway', 'quarry', 'farmyard', 'landfill', 'garages'],
						c.landuseIndustrial,
						'residential',
						c.landuseResidential,
						'retail',
						c.landuseRetail,
						// agriculture group in the VersaTiles Shortbread style
						['brownfield', 'greenfield'],
						c.landuseAgriculture,
						['bare_rock', 'scree', 'shingle'],
						c.landuseConstruction,

						[
							'farmland',
							'vineyard',
							'plant_nursery',
							'orchard',
							'meadow',
							'grassland',
							'grass',
							'heath',
							'swamp',
							'bog',
							'string_bog',
							'wet_meadow',
							'marsh'
						],
						c.landuseNatureLight,
						['forest', 'scrub'],
						c.landuseNatureHeavy,
						['cemetery', 'grave_yard'],
						c.landuseCemetery,
						['beach', 'sand'],
						c.landuseBeach,

						'magenta'
					],
					// soft fade-in like VersaTiles land layers
					'fill-opacity': ['interpolate', ['linear'], ['zoom'], 8, 0.6, 10, 1]
				}
			},
			{
				id: 'sites',
				type: 'fill',
				source: 'osm',
				'source-layer': 'sites',
				// construction is rendered by its own pattern layer below
				filter: ['!=', 'kind', 'construction'],
				paint: {
					'fill-color': [
						'match',
						['get', 'kind'],
						['university', 'school', 'college', 'prison'],
						c.landuseComplex,
						// like VersaTiles: translucent red
						'hospital',
						c.siteHospital,
						'sports_center',
						c.sport,
						'danger_area',
						c.landuseConstruction,
						['parking', 'bicycle_parking'],
						c.pedestrian,

						'magenta'
					],
					'fill-opacity': ['match', ['get', 'kind'], 'hospital', 0.1, 1]
				}
			},
			{
				// like the VersaTiles Shortbread style: faint thin hatching
				id: 'sites-construction',
				type: 'fill',
				source: 'osm',
				'source-layer': 'sites',
				filter: ['==', 'kind', 'construction'],
				paint: {
					'fill-color': c.landuseConstruction,
					'fill-pattern': 'basics:pattern-hatched_thin',
					'fill-opacity': 0.1
				}
			},
			{
				id: 'water',
				type: 'fill',
				source: 'osm',
				'source-layer': 'water_polygons',
				paint: {
					// glaciers white like in VersaTiles
					'fill-color': ['match', ['get', 'kind'], 'glacier', c.glacier, c.water]
				}
			},
			// dams and piers like in VersaTiles
			{
				id: 'dam-area',
				type: 'fill',
				source: 'osm',
				'source-layer': 'dam_polygons',
				paint: {
					'fill-color': c.damPier,
					'fill-opacity': ['interpolate', ['linear'], ['zoom'], 12, 0, 13, 1]
				}
			},
			{
				id: 'dam-line',
				type: 'line',
				source: 'osm',
				'source-layer': 'dam_lines',
				layout: { 'line-cap': 'round', 'line-join': 'round' },
				paint: { 'line-color': c.water }
			},
			{
				id: 'pier-area',
				type: 'fill',
				source: 'osm',
				'source-layer': 'pier_polygons',
				paint: {
					'fill-color': c.damPier,
					'fill-opacity': ['interpolate', ['linear'], ['zoom'], 12, 0, 13, 1]
				}
			},
			{
				id: 'pier-line',
				type: 'line',
				source: 'osm',
				'source-layer': 'pier_lines',
				layout: { 'line-cap': 'round', 'line-join': 'round' },
				paint: { 'line-color': c.damPier }
			},
			{
				id: 'pedestrian',
				type: 'fill',
				source: 'osm',
				'source-layer': 'street_polygons',
				filter: ['in', 'kind', 'pedestrian', 'service'],
				paint: { 'fill-color': c.pedestrian }
			},
			{
				id: 'waterway',
				type: 'line',
				source: 'osm',
				'source-layer': 'water_lines',
				layout: { 'line-cap': 'round', 'line-join': 'round' },
				paint: {
					'line-color': c.water,
					// width hierarchy from VersaTiles: river > canal > stream > ditch
					'line-width': [
						'interpolate',
						['linear'],
						['zoom'],
						9,
						['match', ['get', 'kind'], 'river', 0.5, 'canal', 0.5, 0],
						10,
						['match', ['get', 'kind'], 'river', 3, 'canal', 2, 0],
						14,
						['match', ['get', 'kind'], 'river', 4.6, 'canal', 3.6, 'stream', 1, 0.5],
						15,
						['match', ['get', 'kind'], 'river', 5, 'canal', 4, 'stream', 2, 'ditch', 1, 1],
						17,
						['match', ['get', 'kind'], 'river', 9, 'canal', 8, 'stream', 6, 'ditch', 4, 2],
						18,
						['match', ['get', 'kind'], 'river', 20, 'canal', 17, 'stream', 12, 'ditch', 8, 4],
						20,
						['match', ['get', 'kind'], 'river', 60, 'canal', 50, 'stream', 30, 'ditch', 20, 10]
					]
				}
			},
			// buildings like in VersaTiles: darker footprint + body shifted up-left
			// by 2px for a pseudo-2.5d effect, fading in at z14-15
			{
				id: 'building-outline',
				type: 'fill',
				source: 'osm',
				'source-layer': 'buildings',
				paint: {
					'fill-color': c.buildingOutline,
					'fill-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0, 15, 1]
				}
			},
			{
				id: 'building',
				type: 'fill',
				source: 'osm',
				'source-layer': 'buildings',
				paint: {
					'fill-color': c.building,
					'fill-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0, 15, 1],
					'fill-translate': [-2, -2]
				}
			},
			// bridge decks (man_made=bridge areas) like in VersaTiles
			{
				id: 'bridge-area',
				type: 'fill',
				source: 'osm',
				'source-layer': 'bridges',
				paint: {
					'fill-color': c.bridgeArea,
					'fill-opacity': 0.8
				}
			},
			// casing under major streets on bridges (VersaTiles bridge outlines)
			{
				id: 'bridge-casing',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 12,
				filter: [
					'all',
					['==', ['get', 'bridge'], true],
					['==', ['get', 'rail'], false],
					[
						'!',
						[
							'in',
							['get', 'kind'],
							[
								'literal',
								[
									'footway',
									'track',
									'steps',
									'cycleway',
									'path',
									'unclassified',
									'service',
									'pedestrian'
								]
							]
						]
					],
					[
						'any',
						['!', ['to-boolean', ['get', 'link']]],
						['all', ['==', ['get', 'kind'], 'motorway'], ['>=', ['zoom'], 12]],
						['>=', ['zoom'], 13]
					]
				],
				layout: {
					'line-cap': 'butt'
				},
				paint: {
					'line-color': c.bridgeCasing,
					'line-width': [
						'let',
						'base',
						[
							'case',
							['to-boolean', ['get', 'link']],
							[
								'match',
								['get', 'kind'],
								'motorway',
								3.5,
								'trunk',
								3,
								['primary', 'secondary', 'tertiary'],
								1.75,
								0.0
							],
							[
								'match',
								['get', 'kind'],
								'motorway',
								4,
								'trunk',
								3.5,
								['primary', 'secondary', 'runway', 'taxiway'],
								3,
								'tertiary',
								1.75,
								'residential',
								1.5,
								0.75
							]
						],
						[
							'interpolate',
							['linear'],
							['zoom'],
							12,
							['+', ['*', ['var', 'base'], 1], 2.5],
							16,
							['+', ['*', ['var', 'base'], 4], 2.5],
							20,
							['+', ['*', ['var', 'base'], 8], 2.5]
						]
					]
				}
			},
			{
				id: 'road_back_residential',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: ['==', 'kind', 'residential'],
				layout: {
					'line-cap': 'round'
				},
				paint: {
					'line-color': c.roadBackResidential,
					'line-width': ['interpolate', ['linear'], ['zoom'], 5, 0, 9, 0.5, 12, 1, 16, 4, 20, 20],
					// dimmed in tunnels
					'line-opacity': [
						'interpolate',
						['linear'],
						['zoom'],
						12,
						['case', ['==', ['get', 'tunnel'], true], 0.15, 0.4],
						15,
						['case', ['==', ['get', 'tunnel'], true], 0.4, 1]
					]
				}
			},
			// minor ways (track / living street / service / unclassified) drawn here,
			// below the bigger streets (primary etc.): the big-road bodies paint over
			// them at junctions, while unclassified sits on top of track / living
			// street / service
			// tracks as a thin, grey plain line (no casing), fading in at z13-14
			// so field and forest ways show up early
			{
				id: 'track',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: [
					'all',
					['==', 'kind', 'track'],
					level === 0 ? ['any', ['!has', 'level'], ['==', 'level', level]] : ['==', 'level', level]
				],
				layout: {
					'line-cap': 'round'
				},
				minzoom: 13,
				paint: {
					'line-color': c.track,
					'line-width': [
						'interpolate',
						['linear'],
						['zoom'],
						13,
						0,
						14,
						0.9,
						16,
						1.5,
						18,
						3,
						19,
						4,
						20,
						6
					]
				}
			},
			// living street: a plain solid body like the old service look, no casing
			{
				id: 'living-street',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: [
					'all',
					['==', 'kind', 'living_street'],
					level === 0 ? ['any', ['!has', 'level'], ['==', 'level', level]] : ['==', 'level', level]
				],
				layout: {
					'line-cap': 'round'
				},
				minzoom: 15,
				paint: {
					'line-color': c.livingStreet,
					'line-width': ['interpolate', ['linear'], ['zoom'], 15, 0, 16, 4, 18, 6, 19, 10, 20, 20]
				}
			},
			// service as a plain solid line (no casing), thinner than living street,
			// fading in at z13-14 like tracks and footpaths
			{
				id: 'footway',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: [
					'all',
					['==', 'kind', 'service'],
					level === 0 ? ['any', ['!has', 'level'], ['==', 'level', level]] : ['==', 'level', level]
				],
				layout: {
					'line-cap': 'round'
				},
				minzoom: 13,
				paint: {
					'line-color': c.footway,
					'line-width': [
						'interpolate',
						['linear'],
						['zoom'],
						13,
						0,
						14,
						1,
						16,
						2.5,
						18,
						4,
						19,
						6,
						20,
						12
					]
				}
			},
			// unclassified as a plain borderless line, slightly grey vs the white
			// residential roads
			{
				id: 'unclassified',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: [
					'all',
					['==', 'kind', 'unclassified'],
					level === 0 ? ['any', ['!has', 'level'], ['==', 'level', level]] : ['==', 'level', level]
				],
				layout: {
					'line-cap': 'round'
				},
				minzoom: 14,
				paint: {
					'line-color': c.unclassified,
					'line-width': [
						'interpolate',
						['linear'],
						['zoom'],
						14,
						2,
						15,
						4,
						16,
						6,
						18,
						10,
						19,
						16,
						20,
						24
					]
				}
			},
			{
				id: 'road_back_non_residential',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				// link zoom gating like VersaTiles: motorway_link from z12, other
				// links from z13 (the tiles carry them from z9).
				// minzoom 6 like the road layer: rail first, motorways one zoom later.
				minzoom: 6,
				filter: [
					'all',
					['==', ['get', 'rail'], false],
					[
						'!',
						[
							'in',
							['get', 'kind'],
							[
								'literal',
								[
									'footway',
									'track',
									'steps',
									'cycleway',
									'path',
									'unclassified',
									'residential',
									'service',
									'living_street'
								]
							]
						]
					],
					[
						'any',
						['!', ['to-boolean', ['get', 'link']]],
						['all', ['==', ['get', 'kind'], 'motorway'], ['>=', ['zoom'], 12]],
						['>=', ['zoom'], 13]
					]
				],
				layout: {
					'line-cap': 'round'
				},
				paint: {
					'line-color': c.roadBackNonResidential,
					// dimmed in tunnels
					'line-opacity': ['case', ['==', ['get', 'tunnel'], true], 0.4, 1],
					'line-width': [
						'let',
						'base',
						[
							'case',
							['to-boolean', ['get', 'link']],
							[
								'match',
								['get', 'kind'],
								'motorway',
								3.5,
								'trunk',
								3,
								['primary', 'secondary', 'tertiary'],
								1.75,
								0.0
							],
							[
								'match',
								['get', 'kind'],
								'motorway',
								4,
								'trunk',
								3.5,
								['primary', 'secondary', 'runway', 'taxiway'],
								3,
								'tertiary',
								1.75,
								0.0
							]
						],
						[
							'interpolate',
							['linear'],
							['zoom'],
							5,
							['+', ['*', ['var', 'base'], 0.1], 1],
							9,
							['+', ['*', ['var', 'base'], 0.4], 1],
							12,
							['+', ['*', ['var', 'base'], 1], 1],
							16,
							['+', ['*', ['var', 'base'], 4], 1],
							20,
							['+', ['*', ['var', 'base'], 8], 1]
						]
					]
				}
			},
			{
				id: 'road',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				layout: {
					'line-cap': 'round'
				},
				// like the old release: only roads with a ref get the full-width
				// opaque body here; everything else is carried by the faint back
				// layers (thin at overview zooms) and the minor-way layers.
				// minzoom 6: rail comes first, motorway/trunk one zoom later (only
				// those two kinds exist below z6 in the tiles).
				minzoom: 6,
				filter: [
					'all',
					['has', 'ref'],
					['==', ['get', 'rail'], false],
					[
						'!',
						[
							'in',
							['get', 'kind'],
							[
								'literal',
								[
									'footway',
									'track',
									'steps',
									'cycleway',
									'path',
									'unclassified',
									'service',
									'pedestrian'
								]
							]
						]
					],
					[
						'any',
						['!', ['to-boolean', ['get', 'link']]],
						['all', ['==', ['get', 'kind'], 'motorway'], ['>=', ['zoom'], 12]],
						['>=', ['zoom'], 13]
					],
					[
						'any',
						['==', ['get', 'kind'], 'motorway'],
						['==', ['get', 'kind'], 'trunk'],
						['==', ['get', 'kind'], 'secondary'],
						['>', ['zoom'], 11]
					]
				],
				paint: {
					// dimmed in tunnels
					'line-opacity': ['case', ['==', ['get', 'tunnel'], true], 0.5, 1],
					// like VersaTiles: links get the same color as their parent kind
					// (motorway_link = motorway, ...), only the width is reduced
					'line-color': [
						'match',
						['get', 'kind'],
						'motorway',
						c.motorway,
						'trunk',
						c.motorwayLink,
						['primary', 'secondary', 'runway', 'taxiway'],
						c.primarySecondary,
						'tertiary',
						c.linkTertiary,
						'residential',
						c.residential,
						c.road
					],
					'line-width': [
						'let',
						'base',
						[
							'case',
							['to-boolean', ['get', 'link']],
							[
								'match',
								['get', 'kind'],
								'motorway',
								3,
								'trunk',
								2.5,
								['primary', 'secondary', 'tertiary'],
								1.75,
								0.75
							],
							[
								'match',
								['get', 'kind'],
								'motorway',
								3.5,
								'trunk',
								3,
								['primary', 'secondary', 'runway', 'taxiway'],
								2.5,
								'tertiary',
								1.75,
								'residential',
								1.5,
								0.75
							]
						],
						[
							'interpolate',
							['linear'],
							['zoom'],
							5,
							['*', ['var', 'base'], 0.5],
							9,
							['*', ['var', 'base'], 1],
							12,
							['*', ['var', 'base'], 2],
							16,
							['*', ['var', 'base'], 2.5],
							20,
							['*', ['var', 'base'], 3]
						]
					]
				}
			},
			// streets with designated bicycle infrastructure, tinted like VersaTiles
			{
				id: 'road-bicycle',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 13,
				filter: [
					'all',
					['==', 'bicycle', 'designated'],
					[
						'in',
						'kind',
						'residential',
						'unclassified',
						'service',
						'track',
						'living_street',
						'pedestrian'
					]
				],
				layout: {
					'line-cap': 'round'
				},
				paint: {
					'line-color': c.bicycleRoad,
					'line-width': ['interpolate', ['linear'], ['zoom'], 12, 1, 14, 2, 16, 4, 18, 12, 20, 40]
				}
			},
			// one-way arrows like in VersaTiles
			{
				id: 'marking-oneway',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 16,
				filter: [
					'all',
					['==', 'oneway', true],
					[
						'in',
						'kind',
						'trunk',
						'primary',
						'secondary',
						'tertiary',
						'unclassified',
						'residential',
						'living_street'
					]
				],
				layout: {
					'symbol-placement': 'line',
					'symbol-spacing': 175,
					'icon-rotate': 90,
					'icon-rotation-alignment': 'map',
					'icon-padding': 5,
					'symbol-avoid-edges': true,
					'icon-image': 'basics:marking-arrow'
				},
				paint: {
					'icon-color': c.text,
					'icon-opacity': ['interpolate', ['linear'], ['zoom'], 16, 0, 17, 0.4]
				}
			},
			{
				id: 'marking-oneway-reverse',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 16,
				filter: [
					'all',
					['==', 'oneway_reverse', true],
					[
						'in',
						'kind',
						'trunk',
						'primary',
						'secondary',
						'tertiary',
						'unclassified',
						'residential',
						'living_street'
					]
				],
				layout: {
					'symbol-placement': 'line',
					'symbol-spacing': 75,
					'icon-rotate': -90,
					'icon-rotation-alignment': 'map',
					'icon-padding': 5,
					'symbol-avoid-edges': true,
					'icon-image': 'basics:marking-arrow'
				},
				paint: {
					'icon-color': c.text,
					'icon-opacity': ['interpolate', ['linear'], ['zoom'], 16, 0, 17, 0.4]
				}
			},
			...hillshadeLayers,
			{
				id: 'ferry_routes',
				type: 'line',
				source: 'osm',
				'source-layer': 'ferries',
				paint: {
					'line-width': 1.5,
					'line-dasharray': [2, 3],
					'line-color': c.ferryRoute
				}
			},
			// rail / tram rendering like the VersaTiles Shortbread style: a solid
			// "roadbed" outline with a dashed core on top for heavy rail, thin lines
			// with crosstie dashes for tram-like kinds, own colors for subways.
			{
				id: 'rail-outline',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				// wider and much earlier than VersaTiles (minzoom 8, 1px until z15)
				// so the main rail network already stands out at country level.
				// Fully visible from z5 — one zoom before motorways (their layers
				// have minzoom 6).
				minzoom: 5,
				filter: [
					'all',
					['==', ['get', 'kind'], 'rail'],
					['!', ['has', 'service']],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.railOutline,
					'line-width': ['interpolate', ['linear'], ['zoom'], 5, 1.5, 8, 2, 13, 2.4, 15, 3, 20, 14],
					'line-opacity': ['case', ['==', ['get', 'tunnel'], true], 0.3, 1]
				}
			},
			{
				id: 'lightrail-outline',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 8,
				filter: [
					'all',
					['==', ['get', 'kind'], 'light_rail'],
					['!', ['has', 'service']],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.railOutline,
					'line-width': ['interpolate', ['linear'], ['zoom'], 8, 1, 13, 1, 15, 1, 20, 14],
					'line-opacity': [
						'interpolate',
						['linear'],
						['zoom'],
						11,
						0,
						12,
						['case', ['==', ['get', 'tunnel'], true], 0.5, 1]
					]
				}
			},
			{
				id: 'rail-service-outline',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 14,
				filter: [
					'all',
					['in', ['get', 'kind'], ['literal', ['rail', 'light_rail']]],
					['has', 'service'],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.railOutline,
					'line-width': ['interpolate', ['linear'], ['zoom'], 14, 0, 15, 1, 16, 1, 20, 14],
					'line-opacity': ['case', ['==', ['get', 'tunnel'], true], 0.3, 1]
				}
			},
			{
				id: 'subway-outline',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: [
					'all',
					['==', ['get', 'kind'], 'subway'],
					['!', ['has', 'service']],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.subwayOutline,
					'line-width': [
						'interpolate',
						['linear'],
						['zoom'],
						11,
						0,
						12,
						1,
						15,
						3,
						16,
						3,
						18,
						6,
						19,
						8,
						20,
						10
					],
					'line-opacity': ['interpolate', ['linear'], ['zoom'], 11, 0, 12, 1]
				}
			},
			{
				id: 'tram-outline',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 15,
				filter: [
					'all',
					['in', ['get', 'kind'], ['literal', ['tram', 'narrow_gauge', 'monorail', 'funicular']]],
					['!', ['has', 'service']],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.railOutline,
					'line-width': ['interpolate', ['linear'], ['zoom'], 15, 0, 16, 5, 18, 7, 20, 20],
					'line-dasharray': [0.1, 0.5]
				}
			},
			{
				id: 'rail',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 14,
				filter: [
					'all',
					['in', ['get', 'kind'], ['literal', ['rail', 'light_rail']]],
					['!', ['has', 'service']],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.railCore,
					'line-width': ['interpolate', ['linear'], ['zoom'], 14, 0, 15, 1, 20, 10],
					'line-dasharray': [2, 2],
					'line-opacity': [
						'interpolate',
						['linear'],
						['zoom'],
						14,
						0,
						15,
						['case', ['==', ['get', 'tunnel'], true], 0.3, 1]
					]
				}
			},
			{
				id: 'rail-service',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 15,
				filter: [
					'all',
					['in', ['get', 'kind'], ['literal', ['rail', 'light_rail']]],
					['has', 'service'],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.railCore,
					'line-width': ['interpolate', ['linear'], ['zoom'], 15, 0, 16, 1, 20, 10],
					'line-dasharray': [2, 2],
					'line-opacity': ['case', ['==', ['get', 'tunnel'], true], 0.3, 1]
				}
			},
			{
				id: 'subway',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: [
					'all',
					['==', ['get', 'kind'], 'subway'],
					['!', ['has', 'service']],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.subwayCore,
					'line-width': [
						'interpolate',
						['linear'],
						['zoom'],
						11,
						0,
						12,
						1,
						15,
						2,
						16,
						2,
						18,
						5,
						19,
						6,
						20,
						8
					],
					'line-dasharray': [2, 2],
					'line-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0, 15, 1]
				}
			},
			{
				id: 'tram',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				// like the old release (rail=secondary): visible from z10 at ~1px
				minzoom: 10,
				filter: [
					'all',
					['in', ['get', 'kind'], ['literal', ['tram', 'narrow_gauge', 'monorail', 'funicular']]],
					['!', ['has', 'service']],
					[
						'any',
						['<=', ['zoom'], LEVEL_MIN_ZOOM],
						['!', ['has', 'level']],
						['==', ['get', 'level'], 0],
						['==', ['get', 'level'], level]
					]
				],
				paint: {
					'line-color': c.railOutline,
					'line-width': [
						'interpolate',
						['linear'],
						['zoom'],
						10,
						1.15,
						16,
						1.15,
						17,
						2,
						18,
						3,
						20,
						5
					]
				}
			},
			{
				id: 'aerialway',
				type: 'line',
				source: 'osm',
				'source-layer': 'aerialways',
				paint: {
					'line-color': c.rail,
					'line-dasharray': [10, 2]
				}
			},
			// Indoor overlay: the level-filtered station interior (z17+, plus the
			// platform areas). Drawn on top of the base map so that tracks and roads
			// cannot cut through rooms, platforms, stairs, elevators or their labels.
			{
				id: 'landuse-public-transport',
				type: 'fill',
				source: 'osm',
				'source-layer': 'land',
				filter: [
					'all',
					['==', 'kind', 'public_transport'],
					['any', ['!has', 'level'], ['==', 'level', level]]
				],
				paint: {
					'fill-color': c.publicTransport
				}
			},
			{
				id: 'indoor-corridor',
				type: 'fill',
				source: 'osm',
				'source-layer': 'indoor',
				filter: ['all', ['==', 'indoor', 'corridor'], ['==', 'level', level]],
				paint: {
					'fill-color': c.indoorCorridor,
					'fill-opacity': 0.8
				}
			},
			{
				id: 'indoor',
				type: 'fill',
				source: 'osm',
				'source-layer': 'indoor',
				filter: ['all', ['!in', 'indoor', 'corridor', 'wall', 'elevator'], ['==', 'level', level]],
				paint: {
					'fill-color': c.indoor,
					'fill-opacity': 0.8
				}
			},
			{
				id: 'indoor-outline',
				type: 'line',
				source: 'osm',
				'source-layer': 'indoor',
				filter: ['all', ['!in', 'indoor', 'corridor', 'wall', 'elevator'], ['==', 'level', level]],
				minzoom: 17,
				paint: {
					'line-color': c.indoorOutline,
					'line-width': 2
				}
			},
			// footpaths (highway=footway/path) and pedestrian streets as thin grey
			// dashed lines; cycleways get the same treatment in blue. Staggered like
			// the VersaTiles ladder but earlier: pedestrian fades in z12-13, walking
			// paths (footway, path) and cycleways follow at z13-14. (pedestrian
			// lines are only in the tiles from z13, so they join at z13 at full
			// opacity.)
			{
				id: 'footpath',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				filter: [
					'all',
					['in', 'kind', 'footway', 'path', 'pedestrian', 'cycleway'],
					level === 0 ? ['any', ['!has', 'level'], ['==', 'level', level]] : ['==', 'level', level]
				],
				minzoom: 12,
				paint: {
					'line-color': ['match', ['get', 'kind'], 'cycleway', c.cycleway, c.footpath],
					'line-dasharray': [2, 1],
					'line-opacity': [
						'interpolate',
						['linear'],
						['zoom'],
						12,
						0,
						13,
						['match', ['get', 'kind'], ['pedestrian'], 1, 0],
						14,
						1
					],
					'line-width': ['interpolate', ['linear'], ['zoom'], 12, 0.3, 14, 0.6, 17, 1.8]
				}
			},
			// steps when zoomed out: a thin grey dashed line like normal pedestrian
			// paths, handed off to the ribbon + rung detail below as that fades in
			// around z15-16
			{
				id: 'steps-line',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 12,
				filter: [
					'all',
					['==', 'kind', 'steps'],
					level === 0
						? [
								'any',
								['!has', 'from_level'],
								['any', ['==', 'from_level', level], ['==', 'to_level', level]]
							]
						: ['any', ['==', 'from_level', level], ['==', 'to_level', level]]
				],
				paint: {
					'line-color': c.footpath,
					'line-dasharray': [2, 1],
					'line-opacity': ['interpolate', ['linear'], ['zoom'], 12, 0, 13, 1, 14, 0],
					'line-width': ['interpolate', ['linear'], ['zoom'], 12, 0.3, 13, 0.6, 14, 0.9]
				}
			},
			// steps like the original OSM style: a footway-colored line with short
			// perpendicular tread rungs and no casing / outline
			{
				id: 'steps',
				type: 'line',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 14,
				filter: [
					'all',
					['==', 'kind', 'steps'],
					level === 0
						? [
								'any',
								['!has', 'from_level'],
								['any', ['==', 'from_level', level], ['==', 'to_level', level]]
							]
						: ['any', ['==', 'from_level', level], ['==', 'to_level', level]]
				],
				layout: {
					'line-cap': 'butt'
				},
				paint: {
					'line-color': c.footpath,
					'line-dasharray': [0.4, 0.3],
					'line-width': ['interpolate', ['linear'], ['zoom'], 14, 2, 16, 4.5, 18, 7.5, 20, 13]
				}
			},
			{
				id: 'indoor-elevator-outline',
				type: 'circle',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 17,
				filter: [
					'all',
					['==', 'indoor', 'elevator'],
					['<=', 'from_level', level],
					['>=', 'to_level', level]
				],
				paint: {
					'circle-color': c.elevatorOutline,
					'circle-radius': 16
				}
			},
			{
				id: 'indoor-elevator',
				type: 'circle',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 17,
				filter: [
					'all',
					['==', 'indoor', 'elevator'],
					['<=', 'from_level', level],
					['>=', 'to_level', level]
				],
				paint: {
					'circle-color': c.elevator,
					'circle-radius': 14
				}
			},
			{
				id: 'indoor-elevator-icon',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 17,
				filter: [
					'all',
					['==', 'indoor', 'elevator'],
					['<=', 'from_level', level],
					['>=', 'to_level', level]
				],
				layout: {
					'icon-image': 'elevator',
					'icon-size': 0.9
				}
			},
			{
				id: 'indoor-names',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'indoor',
				minzoom: 17,
				filter: ['any', ['!has', 'level'], ['==', 'level', level]],
				layout: {
					'symbol-placement': 'point',
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Regular'],
					'text-size': 12
				},
				paint: {
					'text-color': c.indoorText,
					'text-halo-color': c.textHalo,
					'text-halo-width': 2,
					'text-halo-blur': 1
				}
			},
			{
				id: 'road-ref-shield',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 6,
				filter: [
					'all',
					['has', 'ref'],
					[
						'any',
						[
							'all',
							['in', ['get', 'kind'], ['literal', ['motorway', 'trunk', 'secondary']]],
							['!', ['to-boolean', ['get', 'link']]]
						],
						['>', ['zoom'], 11]
					]
				],
				layout: {
					'symbol-placement': 'line',
					'text-field': ['get', 'ref'],
					'text-font': ['Noto Sans Regular'],
					'text-size': ['case', ['==', ['get', 'kind'], 'motorway'], 11, 10],
					'text-justify': 'center',
					'text-rotation-alignment': 'viewport',
					'text-pitch-alignment': 'viewport',
					'icon-image': c.shield,
					'icon-text-fit': 'both',
					'icon-text-fit-padding': [0.5, 4, 0.5, 4],
					'icon-rotation-alignment': 'viewport',
					'icon-pitch-alignment': 'viewport'
				},
				paint: {
					'text-color': c.text
				}
			},
			{
				id: 'road-name-text',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'streets',
				minzoom: 14,
				filter: ['==', 'rail', false],
				layout: {
					'symbol-placement': 'line',
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Regular'],
					'text-size': 9
				},
				paint: {
					'text-halo-width': 2,
					'text-halo-blur': 1,
					'text-halo-color': c.textHalo,
					'text-color': c.text
				}
			},
			// house numbers like in VersaTiles: tiny, translucent, z17+
			{
				id: 'housenumbers',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'addresses',
				minzoom: 17,
				filter: ['!=', 'housenumber', ''],
				layout: {
					'symbol-placement': 'point',
					'text-field': ['get', 'housenumber'],
					'text-font': ['Noto Sans Regular'],
					'text-anchor': 'center',
					'text-size': ['interpolate', ['linear'], ['zoom'], 17, 9, 19, 11]
				},
				paint: {
					'text-color': c.housenumber,
					'text-halo-color': c.textHalo,
					'text-halo-width': 1,
					'text-halo-blur': 1
				}
			},
			// lake and river name labels (the profile attaches names directly to
			// the water geometry; Shortbread's *_labels layers are not produced)
			{
				id: 'water-name',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'water_polygons',
				minzoom: 12,
				// way_area thresholds per zoom: ≥8000 m² from z12, ≥2000 from z13,
				// ≥1000 from z14, ≥500 from z15 (zoom brackets: filters cannot nest
				// ['zoom'] inside interpolate)
				filter: [
					'all',
					['!=', ['get', 'name'], ''],
					[
						'any',
						['all', ['<', ['zoom'], 13], ['>=', ['coalesce', ['get', 'way_area'], 0], 8000]],
						[
							'all',
							['>=', ['zoom'], 13],
							['<', ['zoom'], 14],
							['>=', ['coalesce', ['get', 'way_area'], 0], 2000]
						],
						[
							'all',
							['>=', ['zoom'], 14],
							['<', ['zoom'], 15],
							['>=', ['coalesce', ['get', 'way_area'], 0], 1000]
						],
						['all', ['>=', ['zoom'], 15], ['>=', ['coalesce', ['get', 'way_area'], 0], 500]]
					]
				],
				layout: {
					'symbol-placement': 'point',
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Regular'],
					'text-size': ['interpolate', ['linear'], ['zoom'], 11, 10, 16, 14]
				},
				paint: {
					'text-color': c.waterText,
					'text-halo-color': c.textHalo,
					'text-halo-width': 2,
					'text-halo-blur': 1
				}
			},
			{
				id: 'waterway-name',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'water_lines',
				minzoom: 13,
				filter: ['!=', 'name', ''],
				layout: {
					'symbol-placement': 'line',
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Regular'],
					'text-size': 10
				},
				paint: {
					'text-color': c.waterText,
					'text-halo-color': c.textHalo,
					'text-halo-width': 2,
					'text-halo-blur': 1
				}
			},
			{
				id: 'stops-anchor',
				type: 'background',
				layout: { visibility: 'none' },
				paint: {}
			},
			{
				id: 'towns',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'place_labels',
				minzoom: 9,
				// per-kind min zooms as in the VersaTiles Shortbread style; kinds not
				// listed (locality, island, farm, ...) are never rendered there.
				filter: [
					'>=',
					['zoom'],
					[
						'match',
						['get', 'kind'],
						'town',
						9,
						['village', 'suburb'],
						11,
						['hamlet', 'quarter'],
						13,
						'neighbourhood',
						14,
						99
					]
				],
				layout: {
					// "symbol-sort-key": ["get", "population"],
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Regular'],
					'text-size': 12
				},
				paint: {
					'text-halo-width': 2,
					'text-halo-blur': 1,
					'text-halo-color': c.townTextHalo,
					'text-color': c.townText
				}
			},
			{
				id: 'cities',
				type: 'symbol',
				source: 'osm',
				'source-layer': 'place_labels',
				minzoom: 5,
				filter: [
					'>=',
					['zoom'],
					['match', ['get', 'kind'], 'capital', 5, 'state_capital', 6, 'city', 7, 99]
				],
				layout: {
					'symbol-sort-key': ['-', ['coalesce', ['get', 'population'], 0]],
					'text-field': ['get', 'name'],
					'text-font': ['Noto Sans Bold'],
					'text-size': ['interpolate', ['linear'], ['zoom'], 6, 12, 9, 16]
				},
				paint: {
					'text-halo-width': 2,
					'text-halo-blur': 1,
					'text-halo-color': c.citiesTextHalo,
					'text-color': c.citiesText
				}
			},
			{
				id: 'itinerary-anchor',
				type: 'background',
				layout: { visibility: 'none' },
				paint: {}
			}
		]
	};
};
